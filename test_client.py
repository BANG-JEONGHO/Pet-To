import requests

# 로컬에 켜진 FastAPI 서버 주소
url = "http://localhost:8080/api/v1/fit"

# 테스트용 데이터 (실제 인터넷에 있는 강아지/옷 사진 URL 아무거나 넣으셔도 됩니다)
payload = {
    "user_id": "test_user_001",
    "pet_image_url": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?q=80&w=500", # 강아지 사진 예시
    "cloth_image_url": "https://images.unsplash.com/photo-1576566588028-4147f3842f27?q=80&w=500" # 옷 사진 예시
}

print("서버로 가상 피팅 요청을 보냅니다... (1~2분 대기)")
response = requests.post(url, json=payload)

print("응답 상태 코드:", response.status_code)
print("결과 데이터:", response.json())