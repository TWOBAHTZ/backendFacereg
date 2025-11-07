# scripts/train_facebank.py
import os, cv2, numpy as np
from sqlalchemy.orm import Session
from app.db_models import SessionLocal, UserFace
from insightface.app import FaceAnalysis

MEDIA_ROOT = os.getenv("MEDIA_ROOT", "/data/faces/train")

def load_image(path):
    img = cv2.imread(path)
    return img

def main():
    app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection','recognition'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    db: Session = SessionLocal()
    # ดึง path ของรูปทั้งหมดจาก user_faces
    rows = db.query(UserFace.user_id, UserFace.file_path).all()
    db.close()

    # รวมรูปตาม user_id
    imgs_by_user = {}
    for uid, fp in rows:
        full = fp if os.path.isabs(fp) else os.path.join(MEDIA_ROOT, str(uid), os.path.basename(fp))
        imgs_by_user.setdefault(uid, []).append(full)

    facebank = {}  # { user_id: mean_embedding(512,) }
    total_imgs = 0
    for uid, files in imgs_by_user.items():
        embeds = []
        for f in files:
            img = load_image(f)
            if img is None: continue
            faces = app.get(img)
            if not faces: continue
            emb = faces[0].normed_embedding
            emb = emb / np.linalg.norm(emb)
            embeds.append(emb)
            total_imgs += 1
        if embeds:
            facebank[uid] = np.mean(embeds, axis=0).astype(np.float32)

    # save npz
    os.makedirs("./data", exist_ok=True)
    np.savez("./data/facebank.npz", **{str(k): v for k, v in facebank.items()})
    print(f"✅ Facebank updated: users={len(facebank)} images_used={total_imgs}")

if __name__ == "__main__":
    main()
