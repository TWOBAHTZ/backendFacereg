# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends, HTTPException, Response, \
    Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql import func, asc
import os, io, csv, asyncio, base64, time, uuid, shutil  # 👈 เพิ่ม shutil
from datetime import datetime, date, timedelta  # 👈 เพิ่ม timedelta

from fastapi.responses import JSONResponse

from .db_models import get_db, UserFace, User, AttendanceLog, Subject, UserType
from .camera_handler import CameraManager, discover_local_devices
from .ai_engine import refresh_facebank_from_db, load_facebank

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Offline Attendance (Minimal)")

# --- 1. CORS Middleware ---
origins = ["http://localhost:3000", ]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- 2. Mount Static Directories ---
MEDIA_ROOT = os.getenv("MEDIA_ROOT", "./data/faces/train")
# (ลบ COVERS_MEDIA_ROOT และ app.mount ของ /static/covers เพราะ Schema ใหม่ไม่มี cover_image)
os.makedirs(MEDIA_ROOT, exist_ok=True)
app.mount("/static", StaticFiles(directory="data"), name="static")

# --- 3. Camera Manager Setup ---
print("Discovering local devices for initial setup...")
discovered_devices = discover_local_devices(test_frame=False)
available_sources = [d['src'] for d in discovered_devices if d.get('opened', False)]
print(f"Available camera sources found: {available_sources}")

CAMERA_SOURCES = {}
if len(available_sources) > 0:
    CAMERA_SOURCES['entrance'] = available_sources[0]
else:
    CAMERA_SOURCES['entrance'] = "0"

if len(available_sources) > 1:
    CAMERA_SOURCES['exit'] = available_sources[1]
else:
    CAMERA_SOURCES['exit'] = CAMERA_SOURCES['entrance']

print(f"Assigning camera sources: {CAMERA_SOURCES}")
cam_mgr = CameraManager(CAMERA_SOURCES, fps=30, width=640, height=480)


@app.on_event("startup")
async def _startup():
    cnt = load_facebank()
    print(f"[facebank] loaded users={cnt}")


# --- 4. Face Upload & Training Endpoints ---
@app.post("/faces/upload")
async def upload_faces(user_id: int = Form(...), images: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    saved, items = 0, []
    user_dir = os.path.join(MEDIA_ROOT, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    for f in images:
        file_ext = os.path.splitext(f.filename)[1]
        name = f"{uuid.uuid4()}{file_ext}"  # ใช้ UUID ป้องกันชื่อซ้ำ
        dest = os.path.join(user_dir, name)
        content = await f.read()
        with open(dest, "wb") as wf: wf.write(content)
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
        .join(User, User.user_id == User.user_id).all()
    )
    users, total = refresh_facebank_from_db(rows)
    cnt = load_facebank()
    return {"message": "facebank updated", "users": users, "images_used": total, "loaded": cnt}


# --- 5. Camera Control & Stream Endpoints ---
@app.get("/cameras")
def list_cameras(): return {"cams": cam_mgr.list()}


@app.post("/cameras/{cam_id}/open")
def open_camera(cam_id: str):
    cam_mgr.open(cam_id);
    return {"message": f"camera '{cam_id}' opened"}


@app.post("/cameras/{cam_id}/close")
def close_camera(cam_id: str):
    cam_mgr.close(cam_id);
    return {"message": f"camera '{cam_id}' closed"}


@app.get("/cameras/{cam_id}/snapshot", responses={200: {"content": {"image/jpeg": {}}}})
def camera_snapshot(cam_id: str):
    jpg = cam_mgr.get_jpeg(cam_id);
    return Response(content=jpg, media_type="image/jpeg")


@app.get("/cameras/{cam_id}/mjpeg")
def camera_mjpeg(cam_id: str):
    boundary = "frame"

    async def gen():
        if cam_id not in cam_mgr.sources:
            print(f"MJPEG: Camera ID {cam_id} not found in sources.")
            return
        try:
            cam_mgr.open(cam_id)
            print(f"Opening MJPEG stream for {cam_id} (Source: {cam_mgr.sources[cam_id].src})")
            while True:
                try:
                    jpg = cam_mgr.get_jpeg(cam_id)
                    if not jpg:
                        await asyncio.sleep(0.1)
                        continue
                    yield (
                            b"--" + boundary.encode() + b"\r\n" + b"Content-Type: image/jpeg\r\n" + b"Cache-Control: no-cache\r\n" + b"Pragma: no-cache\r\n" + b"Content-Length: " + str(
                        len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except Exception as e:
                    if not cam_mgr.sources[cam_id].is_open:
                        print(f"MJPEG {cam_id} stopping because camera is no longer open.")
                        break
                    await asyncio.sleep(0.1)
                await asyncio.sleep(cam_mgr.interval)
        except Exception as e:
            print(f"Could not open camera {cam_id} for MJPEG: {e}")
        finally:
            print(f"Closing MJPEG stream for {cam_id}")
            cam_mgr.close(cam_id)

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
    if cam_id not in cam_mgr.sources: await ws.close(code=1008, reason="Camera not found"); return
    cam = cam_mgr.sources[cam_id]
    if not cam.is_open:
        try:
            print(f"WS AI opening camera {cam_id}...")
            cam_mgr.open(cam_id)
        except Exception as e:
            await ws.close(code=1011, reason=f"Could not open camera: {e}");
            return
    print(f"[WS AI {cam_id}] Client connected.")
    try:
        while True:
            if not cam.is_open:
                print(f"[WS AI {cam_id}] Camera is closed, disconnecting.")
                break
            results = cam.last_ai_result
            await ws.send_json({"cam_id": cam_id, "results": results, "ai_width": cam_mgr.ai_process_width,
                                "ai_height": cam_mgr.ai_process_height})
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print(f"[WS AI {cam_id}] Client disconnected.")
    except Exception as e:
        print(f"[WS AI {cam_id}] Error: {e}")
    finally:
        print(f"[WS AI {cam_id}] Connection closed.")


@app.get("/cameras/discover")
async def cameras_discover(max_index: int = 10, test_frame: bool = True):
    print("Discovering local devices...")
    active_sources = [c.src for c in cam_mgr.sources.values() if c.is_open]
    print(f"Active sources (will skip test): {active_sources}")
    devs = discover_local_devices(max_index=max_index, test_frame=test_frame, exclude_srcs=active_sources)
    print(f"Discovery found: {devs}")
    return {"devices": devs}


@app.get("/cameras/config")
def get_camera_config(): return {"mapping": {k: v.src for k, v in cam_mgr.sources.items()}}


@app.post("/cameras/config")
def set_camera_config(mapping: dict = Body(..., example={"entrance": "0", "exit": "1"})):
    print(f"Reconfiguring cameras to: {mapping}")
    cam_mgr.reconfigure(mapping)
    return {"message": "camera mapping updated", "mapping": mapping}


# --- 6. Attendance API Endpoints ---
@app.post("/attendance/start")
def start_attendance():
    print("Starting AI processing for all cameras...")
    for cam in cam_mgr.sources.values(): cam.is_ai_paused = False
    return {"message": "Attendance started"}


@app.post("/attendance/stop")
def stop_attendance():
    print("Stopping AI processing for all cameras...")
    for cam in cam_mgr.sources.values(): cam.is_ai_paused = True
    return {"message": "Attendance stopped"}


COOLDOWN_MINUTES = 2  # กันบันทึกซ้ำเร็วเกินไป


@app.get("/attendance/poll", response_model=List[dict])
async def get_attendance_events(db: Session = Depends(get_db)):
    events = cam_mgr.get_attendance_events()
    if not events: return []

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
        event_timestamp = datetime.fromtimestamp(event["timestamp"])

        last_log = db.query(AttendanceLog).filter(
            AttendanceLog.user_id == user_id,
            AttendanceLog.action == event["action"]
        ).order_by(AttendanceLog.timestamp.desc()).first()

        if not last_log or (event_timestamp - last_log.timestamp) > timedelta(minutes=COOLDOWN_MINUTES):
            new_log_db = AttendanceLog(
                user_id=user_id, subject_id=user_info.subject_id, action=event["action"],
                timestamp=event_timestamp, confidence=event.get("confidence")
            )
            db.add(new_log_db);
            db.flush()
            new_logs_for_frontend.append({
                "log_id": new_log_db.log_id, "user_id": user_id,
                "user_name": user_info.name, "student_code": user_info.student_code or "N/A",
                "action": event["action"], "timestamp": event_timestamp.isoformat(),
                "confidence": event.get("confidence")
            })

    if new_logs_for_frontend: db.commit()
    return new_logs_for_frontend


@app.get("/attendance/logs", response_model=List[dict])
async def get_attendance_logs(
        start_date: Optional[date] = None, end_date: Optional[date] = None,
        subject_id: Optional[int] = None, db: Session = Depends(get_db)
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
    query = query.filter(User.is_deleted == 0)  # กรอง User ที่ยังไม่ถูกลบ
    if start_date: query = query.filter(func.date(AttendanceLog.timestamp) >= start_date)
    if end_date: query = query.filter(func.date(AttendanceLog.timestamp) <= end_date)
    if subject_id is not None: query = query.filter(AttendanceLog.subject_id == subject_id)

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


# --- 7. User Management & Subject Endpoints ---

# ✨ Pydantic Models สำหรับ Subject (Schema ใหม่)
class SubjectCreate(BaseModel):
    subject_name: str
    section: Optional[str] = None
    schedule: Optional[str] = None


class SubjectResponse(SubjectCreate):
    subject_id: int

    class Config:  # 👈 เพิ่ม Config.from_attributes (ORM mode)
        from_attributes = True


# Pydantic Models สำหรับ User
class UserCreate(BaseModel):
    student_code: Optional[str] = None
    name: str;
    role: str
    user_type_id: Optional[int] = None
    subject_id: Optional[int] = None
    password_hash: Optional[str] = None


class UserUpdate(BaseModel):
    name: Optional[str] = None
    student_code: Optional[str] = None
    role: Optional[str] = None


# ✨ [แก้ไข] GET /subjects
@app.get("/subjects", response_model=List[SubjectResponse])
def list_subjects(db: Session = Depends(get_db)):
    subjects = db.query(Subject).all()
    return subjects  # 👈 return object Subject ตรงๆ ได้เลยถ้าใช้ ORM mode


# ✨ [แก้ไข] POST /subjects
@app.post("/subjects", response_model=SubjectResponse)
def create_subject(subject: SubjectCreate, db: Session = Depends(get_db)):
    existing = db.query(Subject).filter(
        Subject.subject_name == subject.subject_name,
        Subject.section == subject.section
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Subject with this name and section already exists")

    new_subject = Subject(
        subject_name=subject.subject_name,
        section=subject.section,
        schedule=subject.schedule
    )
    try:
        db.add(new_subject);
        db.commit();
        db.refresh(new_subject)
        return new_subject
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


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
    db.add(user);
    db.commit();
    db.refresh(user)
    return {"message": "User created", "user": {
        "user_id": user.user_id, "student_code": user.student_code,
        "name": user.name, "role": user.role,
        "user_type_id": user.user_type_id, "subject_id": user.subject_id
    }}


@app.get("/users", response_model=List[dict])
def list_users(
        subject_id: Optional[int] = None,
        db: Session = Depends(get_db)
):
    query = db.query(User).options(selectinload(User.faces)).filter(User.is_deleted == 0)
    if subject_id is not None:
        query = query.filter(User.subject_id == subject_id)
    users = query.all()
    results = []
    for u in users:
        results.append({
            "user_id": u.user_id, "student_code": u.student_code,
            "name": u.name, "role": u.role,
            "user_type_id": u.user_type_id, "subject_id": u.subject_id,
            "faces": [
                {"face_id": f.face_id, "file_path": f.file_path}
                for f in u.faces
            ]
        })
    return results


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
        if payload.student_code != user.student_code:
            existing = db.query(User).filter(User.student_code == payload.student_code, User.is_deleted == 0).first()
            if existing:
                raise HTTPException(status_code=400, detail=f"Student code '{payload.student_code}' already exists.")
        user.student_code = payload.student_code
        updated = True
    if updated:
        db.commit()
    return {"message": "User updated", "user_id": user.user_id}


# ✨ [แก้ไข] DELETE /users/{user_id} (เวอร์ชันล่าสุดที่ลบไฟล์และเปลี่ยน student_code)
@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).options(selectinload(User.faces)).filter(User.user_id == user_id,
                                                                   User.is_deleted == 0).first()
    if not user: raise HTTPException(status_code=404, detail="User not found")
    user_face_dir = os.path.join(MEDIA_ROOT, str(user_id))
    try:
        # 1. ลบข้อมูล faces ใน DB
        if user.faces:
            for face in user.faces: db.delete(face)

        # 2. อัปเดต User (Soft delete + แก้ student_code)
        user.is_deleted = 1
        if user.student_code:
            user.student_code = f"{user.student_code}_deleted_{int(time.time())}"

        # 3. Commit การเปลี่ยนแปลงใน DB
        db.commit()

        # 4. ลบโฟลเดอร์รูปภาพออกจาก Disk
        if os.path.isdir(user_face_dir):
            shutil.rmtree(user_face_dir)

    except Exception as e:
        db.rollback();
        raise HTTPException(status_code=500, detail=f"Failed to delete user data: {e}")

    return {"message": f"User {user_id} ({user.name}) marked as deleted and all face data removed."}


@app.delete("/faces/{face_id}")
def delete_face(face_id: int, db: Session = Depends(get_db)):
    face = db.query(UserFace).filter(UserFace.face_id == face_id).first()
    if not face:
        raise HTTPException(status_code=404, detail="Face image not found")
    try:
        file_path = os.path.join(MEDIA_ROOT, str(face.user_id), face.file_path)
        db.delete(face);
        db.commit()
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"message": f"Face image {face_id} ({face.file_path}) deleted."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete face: {e}")


@app.get("/attendance/export")
def export_attendance_logs(
        subject_id: Optional[int] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        db: Session = Depends(get_db)
):
    query = (
        db.query(
            AttendanceLog.user_id,
            User.name.label("user_name"),
            User.student_code,
            AttendanceLog.subject_id,
            Subject.subject_name,
            func.date(AttendanceLog.timestamp).label("log_date"),
            AttendanceLog.action,
            AttendanceLog.timestamp,
        )
        # ✨ [แก้ไขที่ 1] เปลี่ยนจาก .join เป็น .outerjoin
        .outerjoin(User, AttendanceLog.user_id == User.user_id)
        # ✨ (อันนี้ควรแก้แล้วจากรอบก่อน)
        .outerjoin(Subject, AttendanceLog.subject_id == Subject.subject_id)
        # ✨ [แก้ไขที่ 2] ย้าย Filter นี้มาไว้หลัง Join
        # .filter(User.is_deleted == 0)
        .order_by(AttendanceLog.user_id, AttendanceLog.timestamp.asc())
    )

    # ✨ [แก้ไขที่ 3] กรอง is_deleted ตรงนี้แทน
    # (ต้องเช็ค User.is_deleted != 1 เพราะถ้าเป็น NULL (จาก outerjoin) ก็ยังเอา)
    query = query.filter(User.is_deleted != 1)

    if subject_id:
        query = query.filter(AttendanceLog.subject_id == subject_id)
    if start_date:
        query = query.filter(func.date(AttendanceLog.timestamp) >= start_date)
    if end_date:
        query = query.filter(func.date(AttendanceLog.timestamp) <= end_date)

    logs = query.all()

    # จัดกลุ่มตาม user + วัน
    grouped = {}
    for log in logs:
        # ✨ [แก้ไขที่ 4] เผื่อ user_id เป็น None จาก Outer Join
        key = (log.user_id if log.user_id else 'UNKNOWN', log.log_date)
        grouped.setdefault(key, []).append(log)

    results = []
    for (uid, log_date), entries in grouped.items():

        ins = [e for e in entries if e.action.lower() == "enter"]
        outs = [e for e in entries if e.action.lower() == "exit"]

        if not ins:
            continue  # ไม่มีการเข้า

        in_time = ins[0].timestamp
        out_time = outs[-1].timestamp if outs else None

        duration = timedelta(0)
        if out_time:
            duration = out_time - in_time

        user_name = entries[0].user_name if entries[0].user_name else "Unknown User"
        student_code = entries[0].student_code if entries[0].student_code else "N/A"

        results.append({
            "user_id": uid,
            "user_name": user_name,
            "student_code": student_code,
            "subject_id": entries[0].subject_id,
            "subject_name": entries[0].subject_name if entries[0].subject_name else "N/A",
            "date": log_date.isoformat(),
            "in_time": in_time.isoformat(),
            "out_time": out_time.isoformat() if out_time else None,
            "duration_minutes": round(duration.total_seconds() / 60, 2),
            "status": "Present" if out_time else "No Exit"
        })

    return JSONResponse(content=results)


# --- 9. Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)