import base64
import io
import mimetypes

from fastapi import UploadFile


try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

try:
    from PIL import Image
except ImportError:
    Image = None


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_CONTEXT_CHARS = 100_000

ALLOWED_DOCUMENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}
ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/gif",
    "image/bmp",
    "image/webp",
}
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
}


async def process_uploaded_file(file: UploadFile) -> dict:
    if file.size is not None and file.size > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds {MAX_FILE_BYTES} bytes limit")

    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_type = file.content_type or mimetypes.guess_type(filename)[0] or ""

    if ext not in ALLOWED_EXTENSIONS and content_type not in (ALLOWED_DOCUMENT_TYPES | ALLOWED_IMAGE_TYPES):
        raise ValueError("Unsupported file type. Allowed: pdf, docx, doc, txt, png, jpg, jpeg, gif, bmp, webp")

    data = io.BytesIO()
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_BYTES:
            raise ValueError(f"File exceeds {MAX_FILE_BYTES} bytes limit")
        data.write(chunk)
    data.seek(0)

    if content_type in ALLOWED_IMAGE_TYPES or ext in ALLOWED_EXTENSIONS and ext.startswith(".") and ext[1:] in {"png", "jpg", "jpeg", "gif", "bmp", "webp"}:
        return await _process_image(data, filename, ext)

    if ext == ".pdf" or content_type == "application/pdf":
        return _process_pdf(data, filename)

    if ext == ".docx" or content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return _process_docx(data, filename)

    if ext == ".txt" or content_type == "text/plain":
        text = _decompress(data).decode("utf-8", errors="ignore")
        return {"filename": filename, "type": "document", "text": _truncate(text)}

    raise ValueError("Unsupported file type")


def _decompress(buf: io.BytesIO) -> bytes:
    return buf.getvalue()


def _truncate(text: str) -> str:
    if len(text) > MAX_CONTEXT_CHARS:
        return text[:MAX_CONTEXT_CHARS] + "\n...[truncated]"
    return text


def _process_pdf(buf: io.BytesIO, fileName: str) -> dict:
    if pypdf is None:
        raise ImportError("pypdf is not installed. Add 'pypdf' to requirements.")
    reader = pypdf.PdfReader(buf)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n".join(pages)
    return {"filename": fileName, "type": "document", "text": _truncate(text), "pages": len(reader.pages)}


def _process_docx(buf: io.BytesIO, fileName: str) -> dict:
    if docx is None:
        raise ImportError("python-docx is not installed. Add 'python-docx' to requirements.")
    document = docx.Document(buf)
    text = "\n".join(p.text for p in document.paragraphs if p.text)
    return {"filename": fileName, "type": "document", "text": _truncate(text)}


async def _process_image(buf: io.BytesIO, filename: str, ext: str) -> dict:
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    if Image is not None:
        try:
            img = Image.open(buf)
            img.verify()
            buf.seek(0)
        except Exception as exc:
            raise ValueError("Invalid or corrupted image file") from exc
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"filename": filename, "type": "image", "mime_type": mime, "base64": b64}

