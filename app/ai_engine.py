# app/ai_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import cv2
import json
import numpy as np
from typing import Dict, Tuple, List, Optional, Any  # ✨ (ต้องมี Any)
from insightface.app import FaceAnalysis

# ===== Config =====
MEDIA_ROOT = os.getenv("MEDIA_ROOT", "./data/faces/train")
FACEBANK_PATH = os.getenv("FACEBANK_PATH", "./data/facebank.npz")
FACEBANK_NAMES = FACEBANK_PATH + ".names.json"
THRESH = float(os.getenv("RECOG_THRESHOLD", "0.5"))

# ===== Globals =====
_app: Optional[FaceAnalysis] = None
_facebank: Dict[int, np.ndarray] = {}
_usernames: Dict[int, str] = {}


# ===== Utils =====
def _ensure_app() -> FaceAnalysis:
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n != 0 else v


# ===== Public API =====
def load_facebank() -> int:
    global _facebank, _usernames
    if os.path.exists(FACEBANK_PATH):
        data = np.load(FACEBANK_PATH)
        _facebank = {int(k): data[k] for k in data.files}
    else:
        _facebank = {}

    if os.path.exists(FACEBANK_NAMES):
        with open(FACEBANK_NAMES, "r", encoding="utf-8") as f:
            _usernames = {int(k): v for k, v in json.load(f).items()}
    else:
        _usernames = {}
    return len(_facebank)


def refresh_facebank_from_db(rows: List[Tuple[int, str, str]]) -> Tuple[int, int]:
    app = _ensure_app()
    files_by_uid: Dict[int, List[str]] = {}
    names: Dict[int, str] = {}

    for uid, fp, name in rows:
        base = os.path.basename(fp)
        full = fp if os.path.isabs(fp) else os.path.join(MEDIA_ROOT, str(uid), base)
        files_by_uid.setdefault(uid, []).append(full)
        if uid not in names:
            names[uid] = name or f"UID {uid}"

    facebank: Dict[int, np.ndarray] = {}
    total_imgs = 0

    for uid, files in files_by_uid.items():
        embs = []
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            emb = _normalize(faces[0].normed_embedding.astype(np.float32))
            embs.append(emb)
            total_imgs += 1

        if embs:
            facebank[uid] = np.array(embs).astype(np.float32)

    dst_dir = os.path.dirname(FACEBANK_PATH) or "."
    os.makedirs(dst_dir, exist_ok=True)

    np.savez(FACEBANK_PATH, **{str(k): v for k, v in facebank.items()})
    with open(FACEBANK_NAMES, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in names.items()}, f, ensure_ascii=False)

    global _facebank, _usernames
    _facebank = facebank
    _usernames = names
    return len(facebank), total_imgs


def _best_match(emb: np.ndarray) -> Tuple[Optional[int], float]:
    """
    คำนวณ similarity กับทุกคนใน facebank โดยหาค่าที่สูงที่สุด
    """
    if not _facebank:
        return None, -1.0

    best_uid, best_sim = None, -1.0

    for uid, ref_embs in _facebank.items():
        sims = np.dot(ref_embs, emb)
        max_sim = np.max(sims)

        if max_sim > best_sim:
            best_sim, best_uid = max_sim, uid

    return best_uid, float(best_sim)


# ✨✨✨ [ นี่คือฟังก์ชันที่แก้ไข ] ✨✨✨
def annotate_and_match(frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    ค้นหาใบหน้า, เปรียบเทียบ, และคืนค่าเป็น List[dict] ของข้อมูล
    (ไม่วาดลงบน frame)
    """
    app = _ensure_app()
    faces = app.get(frame)
    results: List[Dict[str, Any]] = []  # ✨

    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        emb = _normalize(f.normed_embedding.astype(np.float32))

        uid, sim = _best_match(emb)

        # กำหนดชื่อ
        if uid is not None and sim >= THRESH:
            name = _usernames.get(uid, f"UID {uid}")
            percent = ((sim + 1) / 2) * 100
            display_name = f"{name} ({percent:.1f}%)"
            matched = True
        else:
            name = "Unknown"
            display_name = "Unknown"
            matched = False
            uid = None  # (สำคัญ)

        # คำนวณ w, h
        w = x2 - x1
        h = y2 - y1

        # ✨ สร้าง dict ข้อมูล (ไม่รวม label ที่คำนวณ %)
        result_data = {
            "name": name,
            "box": [int(x1), int(y1), int(w), int(h)],  # ส่งเป็น [x, y, w, h]
            "similarity": float(sim) if sim is not None else None,
            "matched": matched,
            "display_name": display_name,  # (ส่งชื่อที่แสดงผลไปให้ Frontend)
            "user_id": uid  # (ส่ง user_id ไปด้วย)
        }
        results.append(result_data)

        # ❌ ลบส่วนที่วาดออก ❌
        # cv2.rectangle(...)
        # cv2.putText(...)

    return results  # ✅ คืนค่าเป็น List[dict]
# ✨✨✨ --- จบฟังก์ชันที่แก้ไข --- ✨✨✨