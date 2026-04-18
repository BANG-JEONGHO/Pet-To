import cv2
import numpy as np
import torch
import io
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
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
        controlnet = ControlNetModel.from_pretrained(
            "./weights/control_v11p_sd15_inpaint",
            local_files_only=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        self.diffusion_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "./weights/sd-inpainting-model",
            controlnet=controlnet,
            local_files_only=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # IP-Adapter 플러그인 장착 (미리 다운로드 필요)
        self.diffusion_pipe.load_ip_adapter(
            "./weights/IP-Adapter", # 로컬에 다운받을 IP-Adapter 가중치 폴더
            subfolder="models", 
            weight_name="ip-adapter_sd15.bin"
        )

        # 옷 사진(이미지 프롬프트)이 결과물에 미치는 영향력(강도) 조절 (0.0 ~ 1.0)
        self.diffusion_pipe.set_ip_adapter_scale(0.8) 
        
        # 안전 필터 해제 (필요시)
        self.diffusion_pipe.safety_checker = None

        # 3. YOLO 로컬 detector 로드
        self.yolo_model = YOLO("./weights/yolov8n.pt")
        self.bbox_padding_ratio = 0.08
        self.min_confidence = 0.25
        
        print("✅ 모든 AI 모델 로딩 완료!")

    def _get_bounding_box(self, pet_img_bytes: bytes, image_hw: tuple[int, int]) -> np.ndarray:
        """YOLO로 강아지/고양이의 바운딩 박스를 가져옵니다."""
        H, W = image_hw

        img_bgr = cv2.imdecode(np.frombuffer(pet_img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return np.array([0, 0, W - 1, H - 1], dtype=int)

        results = self.yolo_model.predict(
            source=img_bgr,
            conf=self.min_confidence,
            verbose=False,
        )
        if not results:
            return np.array([0, 0, W - 1, H - 1], dtype=int)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.array([0, 0, W - 1, H - 1], dtype=int)

        # COCO 기준: cat=15, dog=16 (커스텀 모델이면 클래스 인덱스에 맞게 변경)
        target_classes = {15, 16}

        best = None
        best_conf = -1.0
        for b in boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            if cls_id in target_classes and conf > best_conf:
                best = b
                best_conf = conf

        if best is None:
            confs = boxes.conf.cpu().numpy()
            best = boxes[int(np.argmax(confs))]

        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().tolist()

        bw = x2 - x1
        bh = y2 - y1
        px = bw * self.bbox_padding_ratio
        py = bh * self.bbox_padding_ratio

        x_min = max(0, int(x1 - px))
        y_min = max(0, int(y1 - py))
        x_max = min(W - 1, int(x2 + px))
        y_max = min(H - 1, int(y2 + py))

        return np.array([x_min, y_min, x_max, y_max], dtype=int)

    def _make_inpaint_condition(self, image_pil: Image.Image, mask_pil: Image.Image) -> torch.Tensor:
        """ControlNet Inpaint용 조건 이미지를 생성합니다."""
        image = np.array(image_pil).astype(np.float32) / 255.0
        mask = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0
        image[mask > 0.5] = -1.0
        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    def generate_fitting(self, pet_img_bytes, cloth_img_bytes):
        # 1. 바이트 이미지를 OpenCV / Numpy 배열로 변환
        pet_img_np = cv2.imdecode(np.frombuffer(pet_img_bytes, np.uint8), cv2.IMREAD_COLOR)
        pet_img_rgb = cv2.cvtColor(pet_img_np, cv2.COLOR_BGR2RGB)
        
        # [STEP 1] YOLO: 위치 감지
        h, w = pet_img_rgb.shape[:2]
        bbox = self._get_bounding_box(pet_img_bytes, (h, w))

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
        
        control_image = self._make_inpaint_condition(pet_pil, mask_pil)

        result_image = self.diffusion_pipe(
            prompt=prompt,
            image=pet_pil,
            mask_image=mask_pil,
            control_image=control_image,
            ip_adapter_image=cloth_pil,
            num_inference_steps=30,
            guidance_scale=7.0,
            controlnet_conditioning_scale=0.9,
            strength=0.9,
            negative_prompt="extra limbs, distorted anatomy, blurry, low quality, watermark, text",
        ).images[0]

        # 최종 합성 이미지를 바이트로 변환하여 반환 (FastAPI에서 전송하기 위함)
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='JPEG')
        return img_byte_arr.getvalue()

# 서버가 켜질 때 딱 한 번만 파이프라인 인스턴스 생성
pipeline = PetVTONPipeline()