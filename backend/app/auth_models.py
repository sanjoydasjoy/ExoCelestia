from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from datetime import datetime
import re


class UserBase(BaseModel):
    """Base user model"""
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, description="User's full name")


class UserCreate(UserBase):
    """Model for user registration"""
    password: str = Field(..., min_length=8, description="User password (min 8 characters)")
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        if not re.search(r'[A-Z]', v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r'[a-z]', v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r'[0-9]', v):
            raise ValueError("Password must contain at least one number")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "astronaut@nasa.gov",
                "full_name": "John Doe",
                "password": "SecurePass123"
            }
        }


class UserLogin(BaseModel):
    """Model for user login"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "astronaut@nasa.gov",
                "password": "SecurePass123"
            }
        }


class User(UserBase):
    """User model with additional fields"""
    id: str
    created_at: datetime
    is_active: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "email": "astronaut@nasa.gov",
                "full_name": "John Doe",
                "created_at": "2024-01-01T00:00:00",
                "is_active": True
            }
        }


class Token(BaseModel):
    """JWT Token response"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    user: User = Field(..., description="User information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "user": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "email": "astronaut@nasa.gov",
                    "full_name": "John Doe",
                    "created_at": "2024-01-01T00:00:00",
                    "is_active": True
                }
            }
        }


class TokenData(BaseModel):
    """Token data for JWT payload"""
    email: Optional[str] = None
    user_id: Optional[str] = None


class UserInDB(User):
    """User model stored in database"""
    hashed_password: str

