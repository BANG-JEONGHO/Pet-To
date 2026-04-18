import cv2
import numpy as np
import torch
import io
from google.cloud import aiplatform
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline,
from PIL import Image

class PetVTONPipeline:
    def __init__(self):
        print("AI 모델 로딩 시작...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. SAM 모델 로드 (로컬 폴더에서)
        sam_checkpoint = "./weights/sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

        # 2. Diffusion 모델 로드 (로컬 폴더에서 오프라인 모드로)
        
        self.diffusion_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "./weights/sd-inpainting-model",
            local_files_only=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # [추가됨] IP-Adapter 플러그인 장착 (미리 다운로드 필요)
        self.diffusion_pipe.load_ip_adapter(
            "./weights/IP-Adapter", # 로컬에 다운받을 IP-Adapter 가중치 폴더
            subfolder="models", 
            weight_name="ip-adapter_sd15.bin"
        )

        # 옷 사진(이미지 프롬프트)이 결과물에 미치는 영향력(강도) 조절 (0.0 ~ 1.0)
        self.diffusion_pipe.set_ip_adapter_scale(0.8) 
        
        # 안전 필터 해제 (필요시)
        self.diffusion_pipe.safety_checker = None

        # 3. GCP AutoML (Vertex AI) 초기화
        aiplatform.init(project="knu-2026-bangjeongho833", location="us-central1")
        self.automl_endpoint = aiplatform.Endpoint("88190178396471296")
        
        print("✅ 모든 AI 모델 로딩 완료!")

    def _get_bounding_box(self, pet_img_bytes):
        """AutoML을 호출하여 강아지/고양이의 바운딩 박스를 가져옵니다."""
        # 실제 연동 시 self.automl_endpoint.predict()를 사용합니다.
        # 지금은 SAM 테스트를 위한 임의의 좌표(x_min, y_min, x_max, y_max)를 반환합니다.
        return np.array([50, 50, 400, 400]) 

    def generate_fitting(self, pet_img_bytes, cloth_img_bytes):
        # 1. 바이트 이미지를 OpenCV / Numpy 배열로 변환
        pet_img_np = cv2.imdecode(np.frombuffer(pet_img_bytes, np.uint8), cv2.IMREAD_COLOR)
        pet_img_rgb = cv2.cvtColor(pet_img_np, cv2.COLOR_BGR2RGB)
        
        # [STEP 1] AutoML: 위치 감지
        bbox = self._get_bounding_box(pet_img_bytes)

        # [STEP 2] SAM: 마스크(실루엣) 추출
        self.sam_predictor.set_image(pet_img_rgb)
        masks, _, _ = self.sam_predictor.predict(
            box=bbox,
            multimask_output=False
        )
        mask_image = masks[0]
        
        # Diffusion에 넣기 위해 PIL 이미지로 변환
        mask_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("L")
        pet_pil = Image.fromarray(pet_img_rgb)

        # [추가됨] 바이트 형태의 옷 사진을 PIL 이미지로 변환
        cloth_img_np = cv2.imdecode(np.frombuffer(cloth_img_bytes, np.uint8), cv2.IMREAD_COLOR)
        cloth_pil = Image.fromarray(cv2.cvtColor(cloth_img_np, cv2.COLOR_BGR2RGB))

        # [STEP 3] Diffusion: 가상 피팅 (Inpainting + IP-Adapter)

        # 텍스트 프롬프트는 단순한 지시만 내립니다.
        prompt = "A pet wearing the target clothing, highly detailed, photorealistic"
        
        result_image = self.diffusion_pipe(
            prompt=prompt,
            image=pet_pil,       # 원본 펫 사진
            mask_image=mask_pil, # SAM이 딴 펫 실루엣 마스크
            ip_adapter_image=cloth_pil, # 핵심! 옷 사진을 그대로 꽂아 넣습니다!
            num_inference_steps=20
        ).images[0]

        # 최종 합성 이미지를 바이트로 변환하여 반환 (FastAPI에서 전송하기 위함)
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()

# 서버가 켜질 때 딱 한 번만 파이프라인 인스턴스 생성
pipeline = PetVTONPipeline()