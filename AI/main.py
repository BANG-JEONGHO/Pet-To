from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import urllib.request
import firebase_admin
from firebase_admin import firestore
from google.cloud import storage
from vton_pipeline import pipeline

# Firebase 초기화 (Cloud Run 환경에서는 자동 인증)
firebase_admin.initialize_app()
db = firestore.client()

def upload_to_gcs(image_bytes, destination_blob_name):
    bucket_name = "pet-fitting-images" # 생성한 버킷 이름
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(image_bytes, content_type='image/jpeg')
    return blob.public_url # 저장된 이미지의 주소 반환

app = FastAPI(title="Pet VTON AI Worker")

# Spring Boot에서 넘어오는 JSON 규격
class FittingRequest(BaseModel):
    user_id: str
    pet_image_url: str
    cloth_image_url: str

def download_image_as_bytes(url: str) -> bytes:
    """URL에서 이미지를 바이트 형태로 다운로드합니다."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        return response.read()

@app.post("/api/v1/fit")
async def process_fitting(req: FittingRequest):
    try:
        # 1. Firestore에 처리 시작 상태 기록
        db.collection('fittings').document(req.user_id).set({
            "status": "processing"
        }, merge=True)

        # 2. 이미지 다운로드 (스토리지 URL -> 바이트)
        pet_bytes = download_image_as_bytes(req.pet_image_url)
        cloth_bytes = download_image_as_bytes(req.cloth_image_url)

        # 3. AI 파이프라인 실행 (무거운 작업)
        result_bytes = pipeline.generate_fitting(pet_bytes, cloth_bytes)

        # 4. 결과 이미지를 GCS에 업로드
        result_url = upload_to_gcs(result_bytes, f"results/{req.user_id}_output.jpg")

        # 5. Firestore에 완료 상태 + 결과 URL 기록
        db.collection('fittings').document(req.user_id).update({
            "status": "done",
            "resultUrl": result_url
        })
        
        return {
            "status": "success",
            "message": "Fitting completed successfully.",
            "result_url": result_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Cloud Run이 서버가 살아있는지 확인하는 엔드포인트"""
    return {"status": "ok"}