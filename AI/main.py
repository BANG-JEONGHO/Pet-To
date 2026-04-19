from fastapi import FastAPI, File, UploadFile, Response
from vton_pipeline import pipeline  # 이미 파일 끝에서 생성된 pipeline 인스턴스를 가져옴
import io

app = FastAPI()

@app.post("/generate-fitting")
async def generate_fitting(
    pet_image: UploadFile = File(...), 
    cloth_image: UploadFile = File(...)
):
    # 1. 업로드된 파일 읽기
    pet_bytes = await pet_image.read()
    cloth_bytes = await cloth_image.read()

    # 2. AI 파이프라인 실행 (vton_pipeline.py의 함수 호출)
    # 정호 님의 모델이 내부적으로 처리 후 바이트를 반환합니다.
    result_bytes = pipeline.generate_fitting(pet_bytes, cloth_bytes)

    # 3. 결과 이미지를 브라우저에 전송
    return Response(content=result_bytes, media_type="image/jpeg")

@app.get("/health")
def check_health():
    return {"status": "ready", "model_loaded": True}