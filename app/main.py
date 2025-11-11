# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends, HTTPException, Response, \
    Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.sql import func
from sqlalchemy import or_
import os, io, csv, asyncio, base64, time, uuid
from datetime import datetime, date
import pandas as pd

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
COVERS_MEDIA_ROOT = "./data/subject_covers"
os.makedirs(MEDIA_ROOT, exist_ok=True)
os.makedirs(COVERS_MEDIA_ROOT, exist_ok=True)
app.mount("/static", StaticFiles(directory="data"), name="static")
app.mount("/static/covers", StaticFiles(directory=COVERS_MEDIA_ROOT), name="static_covers")

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
        name = f"{uuid.uuid4()}{file_ext}"
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
        .join(User, User.user_id == UserFace.user_id).all()
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
        try:
            cam_mgr.open(cam_id)
            print(f"Opening MJPEG stream for {cam_id} (Source: {cam_mgr.sources[cam_id].src})")
            while True:
                try:
                    jpg = cam_mgr.get_jpeg(cam_id)
                    yield (
                            b"--" + boundary.encode() + b"\r\n" + b"Content-Type: image/jpeg\r\n" + b"Cache-Control: no-cache\r\n" + b"Pragma: no-cache\r\n" + b"Content-Length: " + str(
                        len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except Exception as e:
                    print(f"MJPEG gen error for {cam_id}: {e}")
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
            cam_mgr.open(cam_id)
        except Exception as e:
            await ws.close(code=1011, reason=f"Could not open camera: {e}"); return
    print(f"[WS AI {cam_id}] Client connected.")
    try:
        while True:
            results = cam.last_ai_result
            await ws.send_json({"cam_id": cam_id, "results": results, "ai_width": cam_mgr.ai_process_width,
                                "ai_height": cam_mgr.ai_process_height})
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print(f"[WS AI {cam_id}] Client disconnected.")
    except Exception as e:
        print(f"[WS AI {cam_id}] Error: {e}")


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
class ActiveSubjectPayload(BaseModel):
    subject_id: Optional[int] = None


# ✨ [ 1. แก้ไข Endpoint นี้ ]
@app.post("/attendance/set_active_subject")
def set_active_subject(payload: ActiveSubjectPayload, db: Session = Depends(get_db)):
    active_user_ids: Optional[set] = None
    roster_size = 0

    if payload.subject_id is not None:
        users_in_subject = db.query(User.user_id).filter(
            User.subject_id == payload.subject_id,
            User.is_deleted == 0
        ).all()
        active_user_ids = {user.user_id for user in users_in_subject}
        roster_size = len(active_user_ids)
        print(f"[Attendance] Setting active subject {payload.subject_id}. Roster size: {roster_size}")
    else:
        print("[Attendance] Setting active subject to ALL.")
        active_user_ids = None

    # (ส่ง subject_id ไปให้ cam_mgr ด้วย)
    cam_mgr.set_active_roster(active_user_ids, payload.subject_id)

    return {"message": "Active subject updated", "active_subject_id": payload.subject_id, "roster_size": roster_size}


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


# ✨ [ 2. แก้ไข Endpoint นี้ ]
@app.get("/attendance/poll", response_model=List[dict])
async def get_attendance_events(db: Session = Depends(get_db)):
    events = cam_mgr.get_attendance_events()
    if not events: return []
    today = date.today()
    new_logs_for_frontend = []
    user_ids_to_check = {e["user_id"] for e in events if e.get("user_id")}
    if not user_ids_to_check: return []
    users_data = db.query(User.user_id, User.subject_id, User.student_code, User.name).filter(
        User.user_id.in_(user_ids_to_check)).all()
    user_info_map = {u.user_id: u for u in users_data}

    # (ดึง subject_id ที่กำลัง Active จาก cam_mgr)
    active_subject_id = cam_mgr.active_subject_id

    for event in events:
        user_id = event.get("user_id")
        if not user_id or user_id not in user_info_map: continue
        user_info = user_info_map[user_id]

        log_subject_id: Optional[int] = None

        # (Logic ใหม่ในการกำหนด Subject ID)
        if active_subject_id is not None:
            # 1. ถ้ามีวิชา Active (เช่น "SP405") -> บังคับ Log ให้เป็นของวิชานั้น
            log_subject_id = active_subject_id
        else:
            # 2. ถ้าเลือก "All Subjects" -> ให้ใช้วิชาหลักของนักเรียน
            log_subject_id = user_info.subject_id

        existing_log = db.query(AttendanceLog).filter(AttendanceLog.user_id == user_id,
                                                      AttendanceLog.action == event["action"],
                                                      func.date(AttendanceLog.timestamp) == today).first()
        if not existing_log:
            event_timestamp = datetime.fromtimestamp(event["timestamp"])
            new_log_db = AttendanceLog(
                user_id=user_id,
                subject_id=log_subject_id,  # (ใช้ subject_id ที่เราเลือก)
                action=event["action"],
                timestamp=event_timestamp,
                confidence=event.get("confidence")
            )
            db.add(new_log_db);
            db.flush()
            new_log_data = {
                "log_id": new_log_db.log_id, "user_id": user_id,
                "user_name": user_info.name, "student_code": user_info.student_code or "N/A",
                "action": event["action"], "timestamp": event_timestamp.isoformat(),
                "confidence": event.get("confidence"),
                "subject_id": log_subject_id  # (เพิ่มการส่ง subject_id กลับไป)
            }
            new_logs_for_frontend.append(new_log_data)
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
    query = query.filter(User.is_deleted == 0)
    query = query.filter(or_(Subject.is_deleted == None, Subject.is_deleted == 0))

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


@app.get("/attendance/export")
async def export_attendance_logs(
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        subject_id: Optional[int] = None,
        format: str = "csv",
        db: Session = Depends(get_db)
):
    query = (
        db.query(
            AttendanceLog.timestamp.label("Timestamp"),
            User.student_code.label("StudentCode"),
            User.name.label("Name"),
            Subject.subject_name.label("Subject"),
            Subject.section.label("Section"),
            AttendanceLog.action.label("Action"),
            AttendanceLog.confidence.label("Confidence")
        )
        .outerjoin(User, AttendanceLog.user_id == User.user_id)
        .outerjoin(Subject, AttendanceLog.subject_id == Subject.subject_id)
        .order_by(AttendanceLog.timestamp.asc())
    )

    query = query.filter(User.is_deleted == 0)
    query = query.filter(or_(Subject.is_deleted == None, Subject.is_deleted == 0))

    if start_date: query = query.filter(func.date(AttendanceLog.timestamp) >= start_date)
    if end_date: query = query.filter(func.date(AttendanceLog.timestamp) <= end_date)
    if subject_id is not None: query = query.filter(AttendanceLog.subject_id == subject_id)

    logs = query.all()

    if not logs:
        df = pd.DataFrame(columns=["Timestamp", "StudentCode", "Name", "Subject", "Section", "Action", "Confidence"])
        df.loc[0] = ["No data found for the selected filters."] + [""] * 6
    else:
        df = pd.DataFrame([log._asdict() for log in logs])
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(None)

    output = io.BytesIO()
    filename = f"attendance_export_{start_date or 'all'}_to_{end_date or 'all'}"

    if format.lower() == 'xlsx':
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename += ".xlsx"
    else:
        df.to_csv(output, index=False, encoding='utf-8')
        media_type = "text/csv"
        filename += ".csv"

    return Response(
        content=output.getvalue(),
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@app.post("/attendance/clear/{cam_id}")
async def clear_attendance_log(cam_id: str):
    if cam_mgr.clear_attendance_session(cam_id):
        return {"message": f"Attendance session for {cam_id} cleared."}
    else:
        raise HTTPException(status_code=404, detail=f"Camera {cam_id} not found or not open.")


# --- 7. User Management & Subject Endpoints ---
@app.get("/subjects", response_model=List[dict])
def list_subjects(db: Session = Depends(get_db)):
    subjects = db.query(Subject).filter(Subject.is_deleted == 0).all()

    return [
        {"subject_id": s.subject_id, "subject_name": s.subject_name, "section": s.section,
         "cover_image_path": s.cover_image_path, "schedule": s.schedule}
        for s in subjects
    ]


class SubjectCreate(BaseModel):
    subject_name: str
    section: Optional[str] = None
    schedule: Optional[str] = None


@app.post("/subjects")
async def create_subject(
        payload: SubjectCreate,
        db: Session = Depends(get_db)
):
    existing_subject = db.query(Subject).filter(
        Subject.subject_name == payload.subject_name,
        Subject.section == payload.section
    ).first()

    if existing_subject:
        if existing_subject.is_deleted == 1:
            print(f"Undeleting subject: {payload.subject_name}")
            existing_subject.is_deleted = 0
            existing_subject.schedule = payload.schedule
            new_subject = existing_subject
        else:
            print(f"Subject already active: {payload.subject_name}")
            raise HTTPException(status_code=400, detail="Subject with this name and section already exists")
    else:
        print(f"Creating new subject: {payload.subject_name}")
        new_subject = Subject(
            subject_name=payload.subject_name,
            section=payload.section,
            schedule=payload.schedule,
            is_deleted=0
        )
        db.add(new_subject)

    try:
        db.commit()
        db.refresh(new_subject)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    return {
        "subject_id": new_subject.subject_id,
        "subject_name": new_subject.subject_name,
        "section": new_subject.section,
        "schedule": new_subject.schedule,
        "cover_image_path": new_subject.cover_image_path
    }


@app.delete("/subjects/{subject_id}")
def delete_subject(subject_id: int, db: Session = Depends(get_db)):
    subject = db.query(Subject).filter(
        Subject.subject_id == subject_id,
        Subject.is_deleted == 0
    ).first()

    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")

    subject.is_deleted = 1
    db.commit()
    return {"message": f"Subject {subject_id} ({subject.subject_name}) marked as deleted."}


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
    subject_id: Optional[int] = None


@app.post("/users")
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    if payload.role not in ["admin", "operator", "viewer"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    if payload.student_code:
        existing = db.query(User).filter(User.student_code == payload.student_code, User.is_deleted == 0).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Student code '{payload.student_code}' already exists.")
    user = User(
        student_code=payload.student_code,
        name=payload.name,
        role=payload.role,
        user_type_id=payload.user_type_id,
        subject_id=payload.subject_id,
        password_hash=payload.password_hash,
    )
    db.add(user);
    db.commit();
    db.refresh(user)
    return {"message": "User created", "user": {
        "user_id": user.user_id, "student_code": user.student_code,
        "name": user.name, "role": user.role,
        "user_type_id": user.user_type_id,
        "subject_id": user.subject_id
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
            "user_type_id": u.user_type_id,
            "subject_id": u.subject_id,
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

    if payload.subject_id is not None:
        user.subject_id = payload.subject_id if payload.subject_id else None
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
    try:
        train_refresh(db)
    except Exception as e:
        print(f"Warning: train_refresh failed after deleting user: {e}")

    return {"message": f"User {user_id} ({user.name}) marked as deleted."}


@app.delete("/faces/{face_id}")
def delete_face(face_id: int, db: Session = Depends(get_db)):
    face = db.query(UserFace).filter(UserFace.face_id == face_id).first()
    if not face:
        raise HTTPException(status_code=4404, detail="Face image not found")
    try:
        file_path = os.path.join(MEDIA_ROOT, str(face.user_id), face.file_path)
        db.delete(face);
        db.commit()
        if os.path.exists(file_path):
            os.remove(file_path)
        try:
            train_refresh(db)
        except Exception as e:
            print(f"Warning: train_refresh failed after deleting face: {e}")

        return {"message": f"Face image {face_id} ({face.file_path}) deleted."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete face: {e}")


# --- 9. Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)