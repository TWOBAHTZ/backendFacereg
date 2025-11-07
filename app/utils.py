import numpy as np
from typing import List, Tuple, Optional
from .config import settings

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def match_identity(embedding: List[float], db_emb_list: List[Tuple[int, bytes]], decrypt_fn) -> Tuple[Optional[int], float]:
    q = np.asarray(embedding, dtype=np.float32)
    best_id, best_score = None, -1.0
    for uid, enc in db_emb_list:
        raw = decrypt_fn(enc)
        emb = np.frombuffer(raw, dtype=np.float32)
        score = cosine_sim(q, emb)
        if score > best_score:
            best_id, best_score = uid, score
    return best_id, best_score

def decide_action(camera_id: str) -> str:
    return "enter" if camera_id.lower() == "entrance" else "exit"

def flag_from_scores(score: float, liveness: float) -> dict:
    flags = {}
    if score < settings.recog_threshold:
        flags["low_confidence"] = True
    if liveness < settings.anti_spoof_threshold:
        flags["spoofing"] = True
    return flags