# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends, HTTPException, Response, \
    Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session, selectinload  # ✨ 1. เพิ่ม selectinload
from sqlalchemy.sql import func
import os, io, csv, asyncio, base64, time
from datetime import datetime, date

# ✨ 2. เพิ่ม StaticFiles
from fastapi.staticfiles import StaticFiles

from .db_models import get_db, UserFace, User, AttendanceLog, Subject, UserType
from .camera_handler import CameraManager, discover_local_devices
from .ai_engine import refresh_facebank_from_db, load_facebank

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Offline Attendance (Minimal)")

# --- 1. CORS Middleware ---
origins = ["http://localhost:3000", ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✨ 3. Mount Static Directory
# นี่คือการบอกว่า URL ที่ขึ้นต้นด้วย /static ให้ไปดึงไฟล์จากโฟลเดอร์ data
# (สมมติว่าคุณรัน uvicorn จากโฟลเดอร์ root ที่มี app/ และ data/ อยู่)
app.mount("/static", StaticFiles(directory="data"), name="static")

# --- 2. Camera Manager Setup ---
CAMERA_SOURCES = {"entrance": "0", "exit": "1"}
cam_mgr = CameraManager(CAMERA_SOURCES, fps=30, width=640, height=480)


@app.on_event("startup")
async def _startup():
    cnt = load_facebank()
    print(f"[facebank] loaded users={cnt}")


# --- 3. Face Upload & Training Endpoints ---
MEDIA_ROOT = os.getenv("MEDIA_ROOT", "./data/faces/train")
os.makedirs(MEDIA_ROOT, exist_ok=True)


@app.post("/faces/upload")
async def upload_faces(user_id: int = Form(...), images: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    saved, items = 0, []
    user_dir = os.path.join(MEDIA_ROOT, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    for f in images:
        name = os.path.basename(f.filename)
        dest = os.path.join(user_dir, name)
        content = await f.read()
        with open(dest, "wb") as wf:
            wf.write(content)
        uf = UserFace(user_id=user_id, file_path=name)
        db.add(uf)
        items.append({"file": name})
        saved += 1
    db.commit()
    return {"saved": saved, "items": items}


@app.post("/train/refresh")
def train_refresh(db: Session = Depends(get_db)):
    rows = (
        db.query(UserFace.user_id, UserFace.file_path, User.name)
        .join(User, User.user_id == UserFace.user_id)
        .all()
    )
    users, total = refresh_facebank_from_db(rows)
    cnt = load_facebank()
    return {"message": "facebank updated", "users": users, "images_used": total, "loaded": cnt}


# --- 4. Camera Control & MJPEG Stream Endpoints ---
# ( ... โค้ดส่วน /cameras, /mjpeg, /ws/cameras, /discover, /config ... )
# ( ... โค้ดส่วน /ws/ai_results ... )
# ( ... (โค้ดส่วน Attendance API /poll, /logs, /clear) ... )
@app.get("/cameras")
def list_cameras():
    return {"cams": cam_mgr.list()}


@app.post("/cameras/{cam_id}/open")
def open_camera(cam_id: str):
    cam_mgr.open(cam_id)
    return {"message": f"camera '{cam_id}' opened"}


@app.post("/cameras/{cam_id}/close")
def close_camera(cam_id: str):
    cam_mgr.close(cam_id)
    return {"message": f"camera '{cam_id}' closed"}


@app.get("/cameras/{cam_id}/snapshot", responses={200: {"content": {"image/jpeg": {}}}})
def camera_snapshot(cam_id: str):
    jpg = cam_mgr.get_jpeg(cam_id)
    return Response(content=jpg, media_type="image/jpeg")


@app.get("/cameras/{cam_id}/mjpeg")
def camera_mjpeg(cam_id: str):
    boundary = "frame"

    async def gen():
        while True:
            try:
                jpg = cam_mgr.get_jpeg(cam_id)
                yield (
                        b"--" + boundary.encode() + b"\r\n"
                                                    b"Content-Type: image/jpeg\r\n"
                                                    b"Cache-Control: no-cache\r\n"
                                                    b"Pragma: no-cache\r\n"
                                                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                        + jpg + b"\r\n"
                )
            except Exception as e:
                await asyncio.sleep(0.05)
            await asyncio.sleep(cam_mgr.interval)

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}",
                             headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                                      "Connection": "keep-alive"})


@app.websocket("/ws/cameras/{cam_id}")
async def ws_camera(ws: WebSocket, cam_id: str):
    await ws.accept()
    try:
        cam_mgr.open(cam_id)
    except Exception:
        pass
    try:
        while True:
            await asyncio.sleep(0.1)
            try:
                jpg = cam_mgr.get_jpeg(cam_id)
                b64 = base64.b64encode(jpg).decode("ascii")
                await ws.send_json({"type": "frame", "cam_id": cam_id, "data": b64})
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/ai_results/{cam_id}")
async def ws_ai_results(ws: WebSocket, cam_id: str):
    await ws.accept()
    if cam_id not in cam_mgr.sources:
        await ws.close(code=1008, reason="Camera not found")
        return
    cam = cam_mgr.sources[cam_id]
    if not cam.is_open:
        try:
            cam_mgr.open(cam_id)
        except Exception as e:
            await ws.close(code=1011, reason=f"Could not open camera: {e}")
            return
    print(f"[WS AI {cam_id}] Client connected.")
    try:
        while True:
            results = cam.last_ai_result
            await ws.send_json({
                "cam_id": cam_id, "results": results,
                "ai_width": cam_mgr.ai_process_width, "ai_height": cam_mgr.ai_process_height
            })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print(f"[WS AI {cam_id}] Client disconnected.")
    except Exception as e:
        print(f"[WS AI {cam_id}] Error: {e}")


@app.get("/cameras/discover")
async def cameras_discover(max_index: int = 10, test_frame: bool = True):
    open_cams = [c.cam_id for c in cam_mgr.sources.values() if c.is_open]
    print(f"Closing open cameras for discovery: {open_cams}")
    for cam_id in open_cams: cam_mgr.close(cam_id)
    await asyncio.sleep(0.5)
    devs = discover_local_devices(max_index=max_index, test_frame=test_frame)
    print(f"Re-opening cameras: {open_cams}")
    for cam_id in open_cams:
        try:
            cam_mgr.open(cam_id)
        except Exception as e:
            print(f"Could not auto-restart {cam_id}: {e}")
    return {"devices": devs}


@app.get("/cameras/config")
def get_camera_config():
    return {"mapping": {k: v.src for k, v in cam_mgr.sources.items()}}


@app.post("/cameras/config")
def set_camera_config(mapping: dict = Body(..., example={"entrance": "0", "exit": "1"})):
    cam_mgr.reconfigure(mapping)
    for cam_id in mapping.keys():
        try:
            cam_mgr.open(cam_id)
        except Exception as e:
            pass
    return {"message": "camera mapping updated", "mapping": mapping}


@app.get("/attendance/poll", response_model=List[dict])
async def get_attendance_events(db: Session = Depends(get_db)):
    events = cam_mgr.get_attendance_events()
    if not events: return []
    today = date.today()
    new_logs_for_frontend = []
    user_ids_to_check = {e["user_id"] for e in events if e.get("user_id")}
    if not user_ids_to_check: return []
    users_data = db.query(
        User.user_id, User.subject_id, User.student_code, User.name
    ).filter(User.user_id.in_(user_ids_to_check)).all()
    user_info_map = {u.user_id: u for u in users_data}
    for event in events:
        user_id = event.get("user_id")
        if not user_id or user_id not in user_info_map: continue
        user_info = user_info_map[user_id]
        existing_log = db.query(AttendanceLog).filter(
            AttendanceLog.user_id == user_id,
            AttendanceLog.action == event["action"],
            func.date(AttendanceLog.timestamp) == today
        ).first()
        if not existing_log:
            event_timestamp = datetime.fromtimestamp(event["timestamp"])
            new_log_db = AttendanceLog(
                user_id=user_id, subject_id=user_info.subject_id,
                action=event["action"], timestamp=event_timestamp,
                confidence=event.get("confidence")
            )
            db.add(new_log_db)
            db.flush()
            new_log_data = {
                "log_id": new_log_db.log_id, "user_id": user_id,
                "user_name": user_info.name, "student_code": user_info.student_code or "N/A",
                "action": event["action"], "timestamp": event_timestamp.isoformat(),
                "confidence": event.get("confidence")
            }
            new_logs_for_frontend.append(new_log_data)
    if new_logs_for_frontend: db.commit()
    return new_logs_for_frontend


@app.get("/attendance/logs", response_model=List[dict])
async def get_attendance_logs(
        start_date: Optional[date] = None, end_date: Optional[date] = None,
        db: Session = Depends(get_db)
):
    query = (
        db.query(
            AttendanceLog, User.name.label("user_name"),
            User.student_code.label("student_code"), Subject.subject_name.label("subject_name")
        )
        .outerjoin(User, AttendanceLog.user_id == User.user_id)
        .outerjoin(Subject, AttendanceLog.subject_id == Subject.subject_id)
        .order_by(AttendanceLog.timestamp.desc())
    )
    query = query.filter(User.is_deleted == 0)
    if start_date: query = query.filter(func.date(AttendanceLog.timestamp) >= start_date)
    if end_date: query = query.filter(func.date(AttendanceLog.timestamp) <= end_date)
    logs = query.all()
    results = []
    for log, user_name, student_code, subject_name in logs:
        results.append({
            "log_id": log.log_id, "user_id": log.user_id,
            "subject_id": log.subject_id, "action": log.action,
            "timestamp": log.timestamp.isoformat(), "confidence": log.confidence,
            "user_name": user_name or "N/A", "student_code": student_code or "N/A",
            "subject_name": subject_name or None
        })
    return results


@app.post("/attendance/clear/{cam_id}")
async def clear_attendance_log(cam_id: str):
    if cam_mgr.clear_attendance_session(cam_id):
        return {"message": f"Attendance session for {cam_id} cleared."}
    else:
        raise HTTPException(status_code=404, detail=f"Camera {cam_id} not found or not open.")


# --- 8. User Management Endpoints ---
class UserCreate(BaseModel):
    student_code: Optional[str] = None
    name: str
    role: str
    user_type_id: Optional[int] = None
    subject_id: Optional[int] = None
    password_hash: Optional[str] = None


@app.post("/users")
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    if payload.role not in ["admin", "operator", "viewer"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    if payload.student_code:
        existing = db.query(User).filter(User.student_code == payload.student_code, User.is_deleted == 0).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Student code '{payload.student_code}' already exists.")
    user = User(
        student_code=payload.student_code, name=payload.name, role=payload.role,
        user_type_id=payload.user_type_id, subject_id=payload.subject_id,
        password_hash=payload.password_hash,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User created", "user": {
        "user_id": user.user_id, "student_code": user.student_code,
        "name": user.name, "role": user.role,
        "user_type_id": user.user_type_id, "subject_id": user.subject_id
    }}


# ✨✨✨ [ แก้ไข GET /users ] ✨✨✨
@app.get("/users", response_model=List[dict])  # (เพิ่ม response_model)
def list_users(db: Session = Depends(get_db)):
    """
    ดึงรายชื่อผู้ใช้ทั้งหมด (ที่ไม่ถูกลบ) พร้อมรูปภาพ (faces)
    """
    # ใช้ selectinload(User.faces) เพื่อให้ SQLAlchemy ดึงข้อมูล faces มาพร้อมกัน
    users = db.query(User).options(selectinload(User.faces)).filter(User.is_deleted == 0).all()

    results = []
    for u in users:
        results.append({
            "user_id": u.user_id,
            "student_code": u.student_code,
            "name": u.name,
            "role": u.role,
            "user_type_id": u.user_type_id,
            "subject_id": u.subject_id,
            # ✨ ส่ง List ของ faces กลับไปด้วย
            "faces": [
                {"face_id": f.face_id, "file_path": f.file_path}
                for f in u.faces
            ]
        })
    return results


# ✨✨✨ [ เพิ่ม PUT /users/{user_id} ] ✨✨✨
class UserUpdate(BaseModel):
    name: Optional[str] = None
    student_code: Optional[str] = None
    role: Optional[str] = None


@app.put("/users/{user_id}")
def update_user(user_id: int, payload: UserUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == user_id, User.is_deleted == 0).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    updated = False
    if payload.name is not None:
        user.name = payload.name
        updated = True
    if payload.student_code is not None:
        # (เช็คว่า student_code ใหม่ซ้ำกับคนอื่นหรือไม่)
        if payload.student_code != user.student_code:
            existing = db.query(User).filter(User.student_code == payload.student_code, User.is_deleted == 0).first()
            if existing:
                raise HTTPException(status_code=400, detail=f"Student code '{payload.student_code}' already exists.")
        user.student_code = payload.student_code
        updated = True

    if updated:
        db.commit()

    return {"message": "User updated", "user_id": user.user_id}


@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == user_id, User.is_deleted == 0).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_deleted = 1
    db.commit()
    # (เราควรจะสั่ง train ใหม่หลังจากลบ user)
    # train_refresh(db)
    return {"message": f"User {user_id} ({user.name}) marked as deleted."}


# ✨✨✨ [ เพิ่ม DELETE /faces/{face_id} ] ✨✨✨
@app.delete("/faces/{face_id}")
def delete_face(face_id: int, db: Session = Depends(get_db)):
    """
    ลบรูปภาพใบหน้า (ทั้งจาก DB และ File System)
    """
    face = db.query(UserFace).filter(UserFace.face_id == face_id).first()
    if not face:
        raise HTTPException(status_code=404, detail="Face image not found")

    try:
        # สร้าง Path เต็มของไฟล์
        file_path = os.path.join(MEDIA_ROOT, str(face.user_id), face.file_path)

        # ลบไฟล์ออกจาก DB
        db.delete(face)
        db.commit()

        # ลบไฟล์ออกจาก Disk
        if os.path.exists(file_path):
            os.remove(file_path)

        return {"message": f"Face image {face_id} ({face.file_path}) deleted."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete face: {e}")


# --- 9. Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)