# app/api/documents.py

from fastapi import APIRouter, Depends, HTTPException
from bson import ObjectId
from bson.errors import InvalidId

from app.core.security import get_current_user
from app.models.user import TokenData
from app.models.schema import DocumentResponse
from app.database import documents_metadata_collection

router = APIRouter()


@router.get("/{document_id}/status", response_model=DocumentResponse)
async def get_document_status(
    document_id: str,
    current_user: TokenData = Depends(get_current_user),
):
    # Polled by the frontend after an async upload: "processing" -> "ready"/"failed".
    # Scoped to the caller's own documents.
    try:
        oid = ObjectId(document_id)
    except (InvalidId, TypeError):
        raise HTTPException(status_code=404, detail="Document not found")

    doc = await documents_metadata_collection.find_one(
        {"_id": oid, "user_email": current_user.email}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(
        document_id=str(doc["_id"]),
        filename=doc.get("filename", ""),
        status=doc.get("status", "processing"),
        chunk_count=doc.get("chunk_count", 0),
        uploaded_at=doc.get("uploaded_at"),
    )
