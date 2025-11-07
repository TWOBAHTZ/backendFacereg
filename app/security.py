import base64, os, hashlib
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .config import settings

def _load_key() -> bytes:
    if not settings.aes_key_b64:
        return os.urandom(32)
    return base64.b64decode(settings.aes_key_b64)

def aes_encrypt(plaintext: bytes, aad: bytes = b"embeddings") -> bytes:
    key=_load_key(); aesgcm=AESGCM(key); nonce=os.urandom(12)
    ct=aesgcm.encrypt(nonce,plaintext,aad)
    return nonce+ct

def aes_decrypt(ciphertext: bytes, aad: bytes = b"embeddings") -> bytes:
    key=_load_key(); aesgcm=AESGCM(key); nonce,ct=ciphertext[:12],ciphertext[12:]
    return aesgcm.decrypt(nonce,ct,aad)

def hash_password(password: str, salt: bytes = None) -> Tuple[str,str]:
    if salt is None: salt=os.urandom(16)
    dk=hashlib.pbkdf2_hmac("sha256",password.encode(),salt,120000)
    return base64.b64encode(salt).decode(),base64.b64encode(dk).decode()

def verify_password(password: str,salt_b64: str,hash_b64: str) -> bool:
    salt=base64.b64decode(salt_b64)
    dk=hashlib.pbkdf2_hmac("sha256",password.encode(),salt,120000)
    return base64.b64encode(dk).decode()==hash_b64