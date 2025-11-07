# app/config.py
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # ไม่ใส่ default ที่ hardcode URL เพื่อบังคับใช้ .env/ENV
    db_url: str = Field(..., alias="DB_URL")
    aes_key_b64: str = Field(default="", alias="AES_KEY_BASE64")
    recog_threshold: float = Field(default=0.35, alias="RECOG_THRESHOLD")
    anti_spoof_threshold: float = Field(default=0.5, alias="ANTI_SPOOF_THRESHOLD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
