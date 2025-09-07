from pydantic import BaseModel, EmailStr

class UserBase(BaseModel):
    email: EmailStr
    name: str = None

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserInDB(UserBase):
    id: int
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    email: str = None
