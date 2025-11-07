from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class SubjectCreate(BaseModel):
    subject_name: str
    section: Optional[str]
    schedule: Optional[str]

class UserEnroll(BaseModel):
    name: str
    role: str
    subject_id: Optional[int]
    embeddings: Optional[List[float]]