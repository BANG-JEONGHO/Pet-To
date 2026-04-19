import requests

# 1. 주소 수정: 포트 8080 -> 8000, 경로 -> /generate-fitting
url = "http://127.0.0.1:8000/generate-fitting"

# 2. 데이터 방식 수정: JSON이 아닌 실제 이미지 파일을 보냄
# 현재 폴더에 dog.jpg와 cloth.jpg 파일이 있어야 합니다.
files = {
    "pet_image": open("dog.jpg", "rb"),
    "cloth_image": open("cloth.webp", "rb")
}

print("서버로 가상 피팅 요청을 보냅니다... (AI 모델 구동 중)")

try:
    # 3. json=payload 대신 files=files 사용
    response = requests.post(url, files=files)

    if response.status_code == 200:
        with open("result_output.jpg", "wb") as f:
            f.write(response.content)
        print("✅ 피팅 완료! 'result_output.jpg' 파일을 확인하세요.")
    else:
        print(f"❌ 실패 (상태 코드 {response.status_code}):", response.text)
finally:
    files["pet_image"].close()
    files["cloth_image"].close()