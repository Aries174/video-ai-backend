from fastapi import APIRouter, UploadFile, File
import os
import shutil

router = APIRouter()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.get("/test")
def test():
    return {"message": "video route working"}


@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    safe_filename = (
        file.filename
        .replace(" ", "_")
        .replace("#", "")
    )

    file_path = f"{UPLOAD_FOLDER}/{safe_filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": safe_filename,
        "message": "upload success"
    }