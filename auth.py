"""
Authentication module for Attentiveness Tracker.
Handles password hashing, JWT creation/validation, and user operations.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session as DBSession

from config import Config
from db import get_db
from models import User

logger = logging.getLogger(__name__)

# === PASSWORD HASHING ===

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# === JWT ===

security = HTTPBearer(auto_error=False)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=Config.JWT_EXPIRY_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: DBSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency: extracts and validates JWT, returns User.
    Raises 401 if token is missing or invalid.
    """
    if creds is None:
        logger.debug("AUTH: No credentials provided (missing Authorization header)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(creds.credentials, Config.JWT_SECRET_KEY, algorithms=[Config.JWT_ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            logger.warning("AUTH: Token payload missing 'sub' claim")
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = int(sub)
    except (JWTError, ValueError) as e:
        logger.warning(f"AUTH: JWT decode failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        logger.warning(f"AUTH: User id={user_id} not found in database")
        raise HTTPException(status_code=401, detail="User not found")

    logger.debug(f"AUTH: Authenticated user id={user.id} email={user.email}")
    return user


def get_optional_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: DBSession = Depends(get_db),
) -> Optional[User]:
    """
    Same as get_current_user but returns None instead of raising 401.
    Useful for routes that work with or without auth.
    """
    if creds is None:
        return None
    try:
        payload = jwt.decode(creds.credentials, Config.JWT_SECRET_KEY, algorithms=[Config.JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            return None
        return db.query(User).filter(User.id == user_id).first()
    except JWTError:
        return None


# === PYDANTIC SCHEMAS ===

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    email: str
