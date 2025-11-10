# app/camera_handler.py
import cv2
import threading
import time
import platform
import queue
from glob import glob
from typing import Dict, Optional, List, Any, Set
from dataclasses import dataclass, field

from .ai_engine import annotate_and_match


def _backend_flag():
    if platform.system() == "Windows":
        return cv2.CAP_DSHOW
    return 0


@dataclass
class CameraSource:
    cam_id: str
    src: str
    is_open: bool = False
    cap: Optional[cv2.VideoCapture] = None
    last_frame: Optional[bytes] = None
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _stop: bool = field(default=False, repr=False)

    ai_queue: queue.Queue = field(default_factory=queue.Queue, repr=False)
    _ai_thread: Optional[threading.Thread] = field(default=None, repr=False)

    last_ai_result: List[Dict[str, Any]] = field(default_factory=list)
    is_ai_paused: bool = True


class CameraManager:
    def __init__(self, sources: Dict[str, str], fps: int = 144, width: int = 640, height: int = 480):
        self.sources = {k: CameraSource(k, v) for k, v in sources.items()}
        self.interval = 1.0 / max(1, fps)
        self.width = width
        self.height = height
        self.ai_process_width = width
        self.ai_process_height = height

        self.CHECK_IN_DURATION = 1.0
        self.attendance_trackers: Dict[str, Dict[str, float]] = {}
        self.checked_in_session: Dict[str, Set[str]] = {}
        self.check_in_queue = queue.Queue()

    def _open_cap(self, src: str) -> cv2.VideoCapture:
        backend = _backend_flag()
        if src.isdigit():
            cap = cv2.VideoCapture(int(src), backend)
        else:
            cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def open(self, cam_id: str):
        cam = self.sources.get(cam_id)
        if not cam: raise KeyError(f"camera '{cam_id}' not found")
        if cam.is_open: return

        try:
            cam.cap = self._open_cap(cam.src)
            if not cam.cap or not cam.cap.isOpened():
                raise RuntimeError(f"Cannot open camera '{cam_id}' ({cam.src})")
        except Exception as e:
            print(f"Failed to open cap for {cam_id}: {e}")
            self.close(cam_id)  # (ถ้าเปิดไม่สำเร็จ ให้ cleanup)
            raise e  # (ส่ง error ต่อ)

        cam.is_open = True
        cam._stop = False
        cam.ai_queue = queue.Queue(maxsize=1)

        self.attendance_trackers[cam_id] = {}
        self.checked_in_session[cam_id] = set()

        cam._thread = threading.Thread(target=self._loop_stream, args=(cam,), daemon=True)
        cam._thread.start()
        cam._ai_thread = threading.Thread(target=self._loop_ai, args=(cam,), daemon=True)
        cam._ai_thread.start()
        print(f"[CameraManager] Opened camera {cam_id} (Src: {cam.src})")

    # ✨✨✨ [ แก้ไขฟังก์ชันนี้ ] ✨✨✨
    def close(self, cam_id: str):
        cam = self.sources.get(cam_id)
        if not cam: return

        # (ถ้ากล้องปิดอยู่แล้ว ก็ไม่ต้องทำอะไร)
        if not cam.is_open and not cam._thread and not cam._ai_thread:
            return

        print(f"[CameraManager] Closing camera {cam_id}...")
        cam._stop = True

        # 1. ส่งสัญญาณให้ AI thread หยุด (สำคัญมาก)
        if cam.ai_queue:
            try:
                cam.ai_queue.put_nowait(None)
            except queue.Full:
                pass  # (ถ้าคิวเต็ม ก็ไม่เป็นไร)

        # 2. รอให้ Stream thread หยุด (ตัวนี้จะหยุดเร็ว)
        if cam._thread and cam._thread.is_alive():
            cam._thread.join(timeout=1.0)
            if cam._thread.is_alive():
                print(f"[WARN] Stream worker {cam.cam_id} failed to join.")

        # 3. รอให้ AI thread หยุด (ตัวนี้อาจจะช้า)
        if cam._ai_thread and cam._ai_thread.is_alive():
            cam._ai_thread.join(timeout=1.0)
            if cam._ai_thread.is_alive():
                print(f"[WARN] AI worker {cam.cam_id} failed to join.")

        # 4. คืนค่ากล้อง (สำคัญมาก)
        if cam.cap:
            try:
                cam.cap.release()
                print(f"[CameraManager] Released cap for {cam_id}")
            except Exception as e:
                print(f"Error releasing cap for {cam_id}: {e}")

        cam.cap = None
        cam.is_open = False
        cam._thread = None
        cam._ai_thread = None

        self.attendance_trackers.pop(cam_id, None)
        self.checked_in_session.pop(cam_id, None)

        print(f"[CameraManager] Successfully closed camera {cam_id}")

    def _loop_stream(self, cam: CameraSource):
        """Loop 1: สตรีมภาพดิบ (เร็ว)"""
        print(f"[Stream Worker {cam.cam_id}] Started...")
        while not cam._stop and cam.cap and cam.cap.isOpened():
            start_time = time.time()

            # ✨ [แก้ไข] เพิ่มการตรวจสอบ cam.cap อีกชั้น
            if not cam.cap: break

            ok, frame = cam.cap.read()
            if not ok or frame is None:
                print(f"[Stream Worker {cam.cam_id}] Frame read error.")
                time.sleep(0.1)  # พัก 0.1 วิ ถ้าอ่านเฟรมไม่ได้
                continue

            ok2, jpg = cv2.imencode(".jpg", frame)
            if ok2:
                cam.last_frame = jpg.tobytes()
            try:
                cam.ai_queue.put_nowait(frame.copy())
            except queue.Full:
                pass
            processing_time = time.time() - start_time
            sleep_time = self.interval - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        print(f"[Stream Worker {cam.cam_id}] Stopped.")

    def _loop_ai(self, cam: CameraSource):
        """Loop 2: ประมวลผล AI (ช้า) + Logic การเช็คชื่อ"""
        print(f"[AI Worker {cam.cam_id}] Started...")

        trackers = self.attendance_trackers.setdefault(cam.cam_id, {})
        checked_in = self.checked_in_session.setdefault(cam.cam_id, set())
        action_map = {"entrance": "enter", "exit": "exit"}
        current_action = action_map.get(cam.cam_id, cam.cam_id)

        while not cam._stop:
            try:
                # ✨ [แก้ไข] ใช้ timeout เพื่อให้ Loop นี้ไม่ค้างตลอดไป
                frame = cam.ai_queue.get(timeout=1.0)
                if frame is None:
                    break

                if cam.is_ai_paused:
                    cam.last_ai_result = []
                    if trackers:
                        trackers.clear()
                    continue  # (Sleep 1 วิ ไปในตัว)

                ai_results = annotate_and_match(frame)
                cam.last_ai_result = ai_results

                current_time = time.time()
                seen_in_frame = set()

                for res in ai_results:
                    if res.get("matched"):
                        name = res.get("name")
                        user_id = res.get("user_id")
                        if name == "Unknown" or not user_id: continue
                        seen_in_frame.add(name)
                        if name in checked_in: continue

                        if name not in trackers:
                            trackers[name] = current_time
                            print(
                                f"👀 [AI Tracker] '{name}' seen on {cam.cam_id}. Starting {self.CHECK_IN_DURATION}s timer...")
                        else:
                            duration = current_time - trackers[name]
                            if duration >= self.CHECK_IN_DURATION:
                                checked_in.add(name)
                                check_in_data = {
                                    "user_id": user_id, "name": name,
                                    "action": current_action, "timestamp": current_time,
                                    "confidence": res.get("similarity")
                                }
                                self.check_in_queue.put(check_in_data)
                                print(f"✅ [ATTENDANCE] Checked in: {name} (Action: {current_action})")
                                trackers.pop(name, None)

                lost_names = set(trackers.keys()) - seen_in_frame
                for name in lost_names:
                    print(f"❌ [AI Tracker] '{name}' lost on {cam.cam_id}. Resetting timer.")
                    trackers.pop(name, None)

            except queue.Empty:
                # (ถ้าไม่มีเฟรมใหม่ใน 1 วิ ก็วนลูปเช็ค _stop และ is_ai_paused ใหม่)
                pass
            except Exception as e:
                print(f"[AI Worker {cam.cam_id}] Error: {e}")
                cam.last_ai_result = []
                time.sleep(1)

        cam.last_ai_result = []
        print(f"[AI Worker {cam.cam_id}] Stopped.")

    def get_jpeg(self, cam_id: str) -> bytes:
        cam = self.sources.get(cam_id)
        if not cam:
            raise KeyError(f"camera '{cam_id}' not found")
        if not cam.is_open:
            print(f"get_jpeg opening camera {cam_id}...")
            self.open(cam_id)

        t0 = time.time()
        while cam.last_frame is None and time.time() - t0 < 3.0:
            time.sleep(0.05)

        if cam.last_frame is None:
            # ✨ [แก้ไข] อย่าใช้ raise RuntimeError เพราะจะทำให้ mjpeg gen() ล่ม
            # raise RuntimeError(f"Could not get frame from camera '{cam_id}' in time")
            print(f"[WARN] get_jpeg timeout for {cam_id}")
            return b''  # ส่งเฟรมเปล่าแทน
        return cam.last_frame

    def list(self):
        return [{"cam_id": c.cam_id, "src": c.src, "is_open": c.is_open} for c in self.sources.values()]

    def reconfigure(self, new_sources: Dict[str, str]):
        for k in list(self.sources.keys()):
            self.close(k)
        # (รอ 1 วินาที ให้ OS คืนค่ากล้องทั้งหมดก่อน)
        time.sleep(1.0)
        self.sources = {k: CameraSource(k, v) for k, v in new_sources.items()}
        print(f"Reconfigured. New sources: {self.sources}")

    def get_attendance_events(self) -> List[dict]:
        events = []
        while not self.check_in_queue.empty():
            try:
                events.append(self.check_in_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def clear_attendance_session(self, cam_id: str):
        if cam_id in self.checked_in_session:
            self.checked_in_session[cam_id].clear()
            print(f"[INFO] Cleared attendance session for {cam_id}")
            return True
        return False


# (discover_local_devices - เหมือนเดิมจากไฟล์ที่แล้ว)
def discover_local_devices(
        max_index: int = 10,
        test_frame: bool = True,
        exclude_srcs: List[str] = None
) -> List[dict]:
    if exclude_srcs is None:
        exclude_srcs = []
    devices: List[dict] = []
    backend = _backend_flag()
    candidates: List[str] = []
    if platform.system() == "Linux":
        vids = sorted(glob("/dev/video*"))
        for v in vids:
            try:
                idx = str(int(v.split("video")[-1]))
                candidates.append(idx)
            except:
                pass
    else:
        candidates = [str(i) for i in range(max_index)]
    seen = set()
    for src in candidates:
        if src in seen: continue
        seen.add(src)
        if src in exclude_srcs:
            print(f"Skipping discovery for src {src} (already in use).")
            devices.append({
                "src": src, "opened": True, "readable": True,
                "width": 640, "height": 480, "in_use": True
            })
            continue

        cap = cv2.VideoCapture(int(src), backend) if src.isdigit() else cv2.VideoCapture(src)
        opened = cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        ok_frame = False
        if opened and test_frame:
            ok, frame = cap.read()
            ok_frame = bool(ok and frame is not None)
        if cap: cap.release()
        devices.append({
            "src": src, "opened": opened, "readable": ok_frame,
            "width": w, "height": h, "in_use": False
        })
    return devices