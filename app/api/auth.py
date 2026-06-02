from datetime import timedelta
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from app.core.security import authenticate_user, create_access_token, get_password_hash, get_current_user, get_full_user
from app.models.user import Token, UserCreate, User, UserInDB, TokenData
from app.database import users_collection
from app.core.config import settings

router = APIRouter()

@router.get("/me")
async def get_current_user_profile(current_user: TokenData = Depends(get_current_user)):
    user = await get_full_user(current_user)
    return {
        "id": str(user["id"]),
        "email": user["email"],
        "full_name": user.get("full_name"),
        "is_active": user.get("is_active", True),
    }

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register", response_model=User)
async def register_user(user: UserCreate):
    # Check if user already exists
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    user_dict = UserInDB(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        id=str(uuid4())
    )
    
    # Insert into database
    await users_collection.insert_one(user_dict.dict())
    
    # Return user without password
    return User(
        email=user_dict.email,
        full_name=user_dict.full_name,
        id=user_dict.id,
        is_active=user_dict.is_active
    )