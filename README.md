# Pet-VTON (Pet Virtual Try-On)

반려동물(강아지/고양이) 사진에 옷 이미지를 합성하여 입혀주는 가상 피팅(Virtual Try-On) 추론 파이프라인입니다.
`yisol/IDM-VTON`을 베이스 모델로 사용하고, 그 위에 펫 도메인으로 fine-tune된 **delta 가중치**를 덮어씌워 동작합니다.

---

## 1. 한 줄 사용법

```bash
python backend_worker/run_pet_tryon.py \
    --pet_image   test_inputs/my_pet.jpg \
    --cloth_image test_inputs/my_cloth.jpg \
    --output      test_outputs/result.jpg
```

입력 2장(펫 사진 + 옷 사진) → 출력 1장(옷 입은 펫 사진).

---

## 2. 추론 파이프라인

`backend_worker/run_pet_tryon.py` 한 번 실행으로 아래 4단계가 순차 실행됩니다.

| 단계 | 스크립트 | 역할 |
| --- | --- | --- |
| 1. DensePose | `tools/make_animal_densepose.py` | detectron2 + DensePose CSE로 동물 신체 표면 임베딩 생성 |
| 2. SAM 마스크 | `tools/make_pet_gsam_masks.py` | GroundingDINO + SAM(vit_b)로 펫 영역 마스크 |
| 3. 마스크 refine | `tools/refine_pet_agnostic_mask.py` | 얼굴/다리/꼬리 제외, 옷이 입혀질 agnostic 영역만 남김 |
| 4. Delta 추론 | `IDM-VTON/inference_pet_delta.py` | `yisol/IDM-VTON` 베이스 + delta로 try-on 합성 |
| 5. NanoBanana 후처리 | `backend_worker/nanobanana_client.py` | (선택) NanoBanana API로 최종 품질 개선 |

중간 산출물은 `runtime/jobs/<job_id>/` 아래에 저장되고, 최종 결과만 `--output` 경로에 복사됩니다.

### NanoBanana API 연동

```bash
# NanoBanana API 키 설정
export NANOBANANA_API_KEY="your-api-key"

# NanoBanana 후처리 포함 실행
python backend_worker/run_pet_tryon.py \
    --pet_image test_inputs/my_pet.jpg \
    --cloth_image test_inputs/my_cloth.jpg \
    --output test_outputs/result.jpg \
    --use_nanobanana

# dry-run (실제 API 호출 없이 비용만 확인)
python backend_worker/run_pet_tryon.py \
    --pet_image test_inputs/my_pet.jpg \
    --cloth_image test_inputs/my_cloth.jpg \
    --output test_outputs/result.jpg \
    --use_nanobanana --nanobanana_dry_run
```

### 비용 검증

```bash
# 시뮬레이션 (100장 기준, API 호출 없음)
python -m backend_worker.cost_benchmark --mode simulate --num_images 100

# 실제 비교 (API 키 필요)
python -m backend_worker.cost_benchmark --mode live \
    --pet_image test_inputs/my_pet.jpg \
    --cloth_image test_inputs/my_cloth.jpg
```

---

## 3. 폴더 구조

```
pet-vton/
├── README.md                       이 문서
├── env_ai_before_idm_vton.txt      conda env 'ai' pip freeze
│
├── backend_worker/
│   └── run_pet_tryon.py            ★ 추론 진입점 (pet + cloth → result)
│
├── tools/                          ★ 전처리 스크립트
│   ├── make_animal_densepose.py
│   ├── make_pet_gsam_masks.py
│   └── refine_pet_agnostic_mask.py
│
├── IDM-VTON/                       ★ try-on 모델 코드
│   ├── inference_pet_delta.py      delta 추론 엔트리
│   ├── src/                        tryon pipeline / UNet hack
│   ├── ip_adapter/                 IP-Adapter Resampler
│   └── LICENSE.txt
│
├── detectron2/                     ★ DensePose 의존성 (projects/DensePose 사용)
│
├── checkpoints/
│   └── sam_vit_b_01ec64.pth        ★ SAM 가중치 (≈360MB)
│
├── output_pet_vton_delta_101/
│   └── checkpoint-final-140/
│       └── unet_trainable_delta/
│           └── trainable_delta.safetensors   ★ fine-tuned delta (≈330MB)
│
├── test_inputs/                    샘플 입력 (my_pet.jpg, my_cloth.jpg)
├── test_outputs/                   결과 저장용
└── runtime/                        job별 중간 산출물 (자동 생성)
```

---

## 4. 사전 준비

### 4.1 하드웨어 / OS
- Linux (Ubuntu 20.04+ / WSL2 검증)
- NVIDIA GPU, CUDA 12.x, **VRAM 16GB 권장** (FP16 추론)
- 1회 추론 약 40~70초, peak VRAM ≈ 14GB

### 4.2 Python 환경

```bash
conda create -n ai python=3.12 -y
conda activate ai
pip install -r env_ai_before_idm_vton.txt

# detectron2 / DensePose editable 설치
pip install -e ./detectron2
pip install -e ./detectron2/projects/DensePose

# Segment Anything
pip install segment-anything
```

### 4.3 베이스 모델 (HuggingFace, 인터넷 필요)

최초 1회 자동 다운로드되지만 미리 캐싱해두는 것을 권장합니다.

```bash
huggingface-cli download yisol/IDM-VTON
huggingface-cli download IDEA-Research/grounding-dino-base
```

DensePose 동물 CSE 가중치는 추론 중 자동으로 fetch됩니다
(`https://dl.fbaipublicfiles.com/densepose/cse/.../model_final_421d28.pkl`).

---

## 5. 옵션 / 인자

`run_pet_tryon.py` 주요 인자:

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--pet_image` | (필수) | 반려동물 이미지 경로 |
| `--cloth_image` | (필수) | 옷 이미지 경로 |
| `--output` | (필수) | 결과 저장 경로 |
| `--pet_type` | `auto` | `auto` / `dog` / `cat` |
| `--item_name` | `pet sweater` | 옷 카테고리 텍스트 (태그용) |
| `--num_inference_steps` | `30` | Diffusion step 수 |
| `--guidance_scale` | `2.0` | CFG scale |
| `--seed` | `42` | 재현성 seed |
| `--delta_checkpoint` | `output_pet_vton_delta_101/checkpoint-final-140` | fine-tuned delta 폴더 |
| `--project_root` | `~/projects/pet-vton` | 프로젝트 루트 |

---

## 6. 입력 가이드 / 알려진 제약

- 펫만 크게 찍힌 단일 펫 사진 권장 (사람이 같이 있으면 DensePose가 사람을 우선 잡을 수 있음)
- 매우 어두운 배경/저해상도 입력은 GroundingDINO가 펫을 못 잡을 수 있음
- 옷(cloth) 이미지는 평평한 ghost mannequin / flat-lay 스타일이 가장 잘 동작
- `--pet_type auto`는 dog/cat 모두 시도하므로 종이 확실하면 명시하는 편이 안정적
- 내부에서 768×1024로 자동 리사이즈됨 (너무 큰 원본은 사전 리사이즈 권장)

---

## 7. 결과물 위치

- 최종 결과: `--output`으로 지정한 경로
- 디버그용 중간 산출물: `runtime/jobs/<job_id>/`
  - `raw_pet_images/`, `raw_cloth_images/`
  - `processed/image-densepose/`, `processed/agnostic-mask/`, `processed/agnostic-mask-refined/`, `processed/agnostic-preview-refined/`
  - `dataset/test/` (IDM-VTON 포맷)
  - `infer_output/` (모델 raw output)

`runtime/jobs/<id>/`는 누적되므로 운영 환경에선 결과 복사 후 주기적으로 정리하세요.

---

## 8. 라이선스

- 베이스 모델 `yisol/IDM-VTON`: 원본 라이선스 따름 (`IDM-VTON/LICENSE.txt`)
- 학습된 delta (`output_pet_vton_delta_101/`): 내부 자산
