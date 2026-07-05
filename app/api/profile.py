# app/api/profile.py
#
# The profile endpoints — how a user's resume becomes their known stack WITHOUT
# ever touching the RAG pipeline. Upload once, and every chat knows the stack.

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException

from app.core.security import get_current_user
from app.models.user import TokenData
from app.services.file_service import process_uploaded_file
from app.services.profile_service import (
    extract_stack,
    save_user_profile,
    get_user_profile,
    delete_user_profile,
)

router = APIRouter()


@router.post("/resume")
async def upload_resume(
    file: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_user),
):
    # Extract the user's tech stack from a resume and store it on their profile.
    # NOT indexed for RAG — a resume is standing context about the user, not
    # query-driven reference material.
    result = await process_uploaded_file(file)

    if result.get("type") != "document":
        raise HTTPException(
            status_code=400,
            detail="Please upload a resume as a PDF, DOCX, or TXT document.",
        )

    text = result.get("text", "")
    extracted = await extract_stack(text)

    if not extracted["is_profile"] or not extracted["known_stack"]:
        # Guard against dropping study material into the resume slot.
        raise HTTPException(
            status_code=422,
            detail="That doesn't look like a resume — we couldn't find a tech "
            "stack to extract. Upload a CV/resume, or just tell me what you know in chat.",
        )

    profile = await save_user_profile(
        user_email=current_user.email,
        known_stack=extracted["known_stack"],
        resume_filename=result.get("filename"),
    )

    return {
        "known_stack": profile["known_stack"],
        "resume_filename": profile.get("resume_filename"),
        "updated_at": profile.get("updated_at"),
    }


@router.get("")
async def read_profile(current_user: TokenData = Depends(get_current_user)):
    profile = await get_user_profile(current_user.email)
    return {
        "known_stack": profile.get("known_stack"),
        "resume_filename": profile.get("resume_filename"),
        "updated_at": profile.get("updated_at"),
    }


@router.delete("")
async def clear_profile(current_user: TokenData = Depends(get_current_user)):
    await delete_user_profile(current_user.email)
    return {"status": "ok", "message": "Profile cleared"}
