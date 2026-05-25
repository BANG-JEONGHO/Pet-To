"""
나노바나나 API만으로 pet virtual try-on을 수행하는 스크립트.
로컬 전처리(DensePose, SAM, IDM-VTON) 없이 나노바나나에 모든 작업을 맡긴다.

현실적 비용 모델:
  - 사전 마스크/포즈 정보가 없으므로 범용 프롬프트로 시도
  - 1차 결과가 불만족스러우면 프롬프트를 바꿔 재시도 (최대 3회)
  - 어디에 옷을 입혀야 하는지 모르므로 얼굴/꼬리에 입히는 오류 발생 가능 → 보정 호출

비교 목적:
  - 이 스크립트(A): 나노바나나만 사용 → 재시도 포함 비용 높음
  - run_pet_tryon.py --use_nanobanana(B): 로컬 전처리 + 품질게이트 → 비용 낮음

사용법:
  export NANOBANANA_API_KEY="your-key"
  python backend_worker/run_nanobanana_only.py \
      --pet_image test_inputs/my_pet.jpg \
      --cloth_image test_inputs/my_cloth.jpg \
      --output test_outputs/result_nanobanana_only.jpg
"""

import argparse
import json
import uuid
from pathlib import Path

from backend_worker.nanobanana_client import (
    COST_PER_IMAGE_EDIT,
    NanoBananaUsage,
    full_tryon_nanobanana,
)


# 나노바나나만 사용할 때의 프롬프트 시퀀스
# 마스크/포즈 정보 없이 범용 프롬프트 → 실패 시 점점 구체화
PROMPT_SEQUENCE = [
    # 1차 시도: 범용 프롬프트 (마스크 없이는 위치 지정 불가)
    (
        "Make this pet wear the clothing item shown in the second image. "
        "The clothing should naturally fit the pet's body. "
        "Output a photorealistic result."
    ),
    # 2차 시도: 위치 보정 프롬프트 (얼굴/꼬리 오류 수정)
    (
        "The pet is wearing clothes incorrectly. "
        "Fix the image so the clothing covers only the torso/body area. "
        "The pet's face, head, ears, paws, and tail must be fully visible and uncovered. "
        "Make it look natural and photorealistic."
    ),
    # 3차 시도: 텍스처 보정 프롬프트
    (
        "Enhance this pet clothing photo: fix any unnatural edges between the clothing "
        "and the pet's fur. Improve the clothing texture to look realistic. "
        "Keep the pet's face and pose unchanged."
    ),
]


def simulate_nanobanana_only_cost(num_attempts: int = 3) -> NanoBananaUsage:
    """
    나노바나나만 사용 시 현실적인 비용 시뮬레이션.

    마스크/포즈 정보가 없으므로:
    - 1차: 기본 try-on 시도 (항상 호출)
    - 2차: 위치 보정 (약 70% 확률 필요 - 얼굴/꼬리에 옷 입히는 오류)
    - 3차: 텍스처 보정 (약 40% 확률 필요)
    → 평균 2.1회 호출
    """
    usage = NanoBananaUsage(mode="full_nanobanana")

    for i in range(num_attempts):
        usage.add_call(
            action=f"tryon_attempt_{i + 1}",
            cost_per_image=COST_PER_IMAGE_EDIT,
            count=1,
        )

    return usage


def main():
    parser = argparse.ArgumentParser(
        description="나노바나나 API만으로 pet try-on (비교용 baseline)"
    )
    parser.add_argument("--pet_image", required=True, help="반려동물 이미지 경로")
    parser.add_argument("--cloth_image", required=True, help="옷 이미지 경로")
    parser.add_argument("--output", required=True, help="결과 저장 경로")
    parser.add_argument("--max_attempts", type=int, default=3,
                        help="최대 시도 횟수 (기본 3회)")
    parser.add_argument("--dry_run", action="store_true",
                        help="실제 API 호출 없이 비용 시뮬레이션만 수행")
    parser.add_argument("--project_root",
                        default=str(Path.home() / "projects" / "pet-vton"))

    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    job_id = uuid.uuid4().hex[:12]
    job_root = project_root / "runtime" / "jobs_nanobanana_only" / job_id
    job_root.mkdir(parents=True, exist_ok=True)

    usage = NanoBananaUsage(mode="full_nanobanana")
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[NANOBANANA-ONLY] dry-run: 비용 시뮬레이션")
        print(f"[NANOBANANA-ONLY] 마스크/포즈 정보 없이 {args.max_attempts}회 시도 가정")
        usage = simulate_nanobanana_only_cost(num_attempts=args.max_attempts)
    else:
        # 실제 API 호출: 프롬프트 시퀀스를 순차 시도
        for i, prompt in enumerate(PROMPT_SEQUENCE[:args.max_attempts]):
            attempt_output = job_root / f"attempt_{i + 1}.jpg"
            print(f"\n[NANOBANANA-ONLY] 시도 {i + 1}/{args.max_attempts}...")
            print(f"[NANOBANANA-ONLY] 프롬프트: {prompt[:80]}...")

            full_tryon_nanobanana(
                pet_image_path=args.pet_image,
                cloth_image_path=args.cloth_image,
                output_path=str(attempt_output),
                prompt=prompt,
                usage=usage,
            )

        # 마지막 시도 결과를 최종 출력으로 사용
        last_attempt = job_root / f"attempt_{min(args.max_attempts, len(PROMPT_SEQUENCE))}.jpg"
        if last_attempt.exists():
            import shutil
            shutil.copy2(last_attempt, output_path)
            print(f"\n[NANOBANANA-ONLY] 최종 결과: {output_path}")

    # 사용량 로그 저장
    cost_log = usage.summary()
    cost_log["note"] = (
        "마스크/포즈 정보 없이 범용 프롬프트로 시도. "
        f"총 {usage.api_calls}회 API 호출 필요. "
        "로컬 전처리 파이프라인이 있으면 품질게이트를 통해 0~1회로 줄일 수 있음."
    )
    cost_log_path = job_root / "nanobanana_usage.json"
    cost_log_path.write_text(
        json.dumps(cost_log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n[NANOBANANA-ONLY] 사용량 로그: {cost_log_path}")
    print(json.dumps(cost_log, indent=2))
    print(f"\n[DONE] job_id: {job_id}")


if __name__ == "__main__":
    main()
