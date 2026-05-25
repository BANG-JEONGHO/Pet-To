"""
비용 검증 스크립트: 로컬 파이프라인 + NanoBanana vs 순수 NanoBanana 비교.

사용법:
  # 시뮬레이션 모드 (API 호출 없이 비용만 비교)
  python -m backend_worker.cost_benchmark --mode simulate --num_images 100

  # 실제 API 호출로 1장 비교 (NANOBANANA_API_KEY 필요)
  python -m backend_worker.cost_benchmark --mode live \
      --pet_image test_inputs/my_pet.jpg \
      --cloth_image test_inputs/my_cloth.jpg
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from backend_worker.nanobanana_client import (
    COST_PER_IMAGE_EDIT,
    NanoBananaUsage,
    full_tryon_nanobanana,
    refine_with_nanobanana,
    simulate_full_tryon,
    simulate_refine,
)


# ──────────────────────────────────────────────
#  비용 모델 상수
# ──────────────────────────────────────────────

# NanoBanana 순수 사용 시 평균 API 호출 수 (try-on 1건당)
# - 마스크/포즈 정보 없이 범용 프롬프트로 시도
# - 1차: 기본 try-on (항상)
# - 2차: 위치 보정 (얼굴/꼬리에 옷 입히는 오류, ~70% 확률)
# - 3차: 텍스처 보정 (~40% 확률)
# → 가중 평균: 1 + 0.7 + 0.4 = 2.1회
AVG_FULL_NANOBANANA_CALLS_PER_IMAGE = 2.1

# 로컬 파이프라인 + 품질 게이트 기반 NanoBanana API 호출 수
# - 품질 게이트 SKIP (IDM-VTON 결과 충분): ~50% → 0회
# - 품질 게이트 REFINE (보정 필요): ~40% → 1회
# - 품질 게이트 REJECT (입력 불량): ~10% → 0회 (호출 차단)
# → 가중 평균: 0.5*0 + 0.4*1 + 0.1*0 = 0.4회
AVG_LOCAL_PIPELINE_CALLS_PER_IMAGE = 0.4

# 로컬 GPU 추론 비용 (전기세 + GPU 감가상각 기준)
# - A100 80GB 기준 시간당 ~$2.21 (AWS p4d.24xlarge / 8 GPU 중 1 GPU)
# - 1회 추론 약 60초 → $0.037
# - RTX 4090 기준 시간당 ~$0.50 → 1회 추론 약 $0.008
LOCAL_GPU_COST_PER_INFERENCE = 0.01  # 보수적 추정


@dataclass
class BenchmarkResult:
    scenario: str
    num_images: int
    total_api_calls: int
    total_nanobanana_cost: float
    local_compute_cost: float
    total_cost: float

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "num_images": self.num_images,
            "total_api_calls": self.total_api_calls,
            "total_nanobanana_cost": round(self.total_nanobanana_cost, 4),
            "local_compute_cost": round(self.local_compute_cost, 4),
            "total_cost": round(self.total_cost, 4),
            "cost_per_image": round(self.total_cost / max(self.num_images, 1), 4),
        }


def estimate_full_nanobanana(num_images: int) -> BenchmarkResult:
    """순수 NanoBanana API만 사용하는 시나리오."""
    total_calls = int(num_images * AVG_FULL_NANOBANANA_CALLS_PER_IMAGE)
    api_cost = total_calls * COST_PER_IMAGE_EDIT

    return BenchmarkResult(
        scenario="A. 순수 NanoBanana (로컬 전처리 없음)",
        num_images=num_images,
        total_api_calls=total_calls,
        total_nanobanana_cost=api_cost,
        local_compute_cost=0.0,
        total_cost=api_cost,
    )


def estimate_local_plus_refine(num_images: int) -> BenchmarkResult:
    """로컬 파이프라인 + NanoBanana refinement 시나리오."""
    total_calls = int(num_images * AVG_LOCAL_PIPELINE_CALLS_PER_IMAGE)
    api_cost = total_calls * COST_PER_IMAGE_EDIT
    local_cost = num_images * LOCAL_GPU_COST_PER_INFERENCE

    return BenchmarkResult(
        scenario="B. 로컬 파이프라인 + NanoBanana 후처리",
        num_images=num_images,
        total_api_calls=total_calls,
        total_nanobanana_cost=api_cost,
        local_compute_cost=local_cost,
        total_cost=api_cost + local_cost,
    )


def estimate_local_only(num_images: int) -> BenchmarkResult:
    """로컬 파이프라인만 사용 (NanoBanana 없음)."""
    local_cost = num_images * LOCAL_GPU_COST_PER_INFERENCE

    return BenchmarkResult(
        scenario="C. 로컬 파이프라인만 (NanoBanana 없음, 참고용)",
        num_images=num_images,
        total_api_calls=0,
        total_nanobanana_cost=0.0,
        local_compute_cost=local_cost,
        total_cost=local_cost,
    )


def run_simulation(num_images: int) -> dict:
    """비용 시뮬레이션 (API 호출 없음)."""
    result_a = estimate_full_nanobanana(num_images)
    result_b = estimate_local_plus_refine(num_images)
    result_c = estimate_local_only(num_images)

    saving_abs = result_a.total_cost - result_b.total_cost
    saving_pct = (saving_abs / result_a.total_cost * 100) if result_a.total_cost > 0 else 0
    api_call_reduction = result_a.total_api_calls - result_b.total_api_calls
    api_call_reduction_pct = (
        (api_call_reduction / result_a.total_api_calls * 100)
        if result_a.total_api_calls > 0
        else 0
    )

    report = {
        "benchmark_type": "simulation",
        "num_images": num_images,
        "assumptions": {
            "nanobanana_cost_per_image_edit": COST_PER_IMAGE_EDIT,
            "avg_full_nanobanana_calls_per_tryon": AVG_FULL_NANOBANANA_CALLS_PER_IMAGE,
            "avg_local_pipeline_calls_per_tryon": AVG_LOCAL_PIPELINE_CALLS_PER_IMAGE,
            "local_gpu_cost_per_inference": LOCAL_GPU_COST_PER_INFERENCE,
            "quality_gate_distribution": {
                "SKIP_ratio": 0.50,
                "SKIP_description": "IDM-VTON 결과 품질 충분 → API 0회",
                "REFINE_ratio": 0.40,
                "REFINE_description": "보정 필요 → API 1회 (마스크 기반 스마트 프롬프트)",
                "REJECT_ratio": 0.10,
                "REJECT_description": "입력 불량 → API 차단 0회 (헛돈 방지)",
            },
            "nanobanana_only_retry_reason": (
                "마스크/포즈 정보 없이 범용 프롬프트 → "
                "위치 오류(얼굴/꼬리에 옷) 70% 확률 재시도 + "
                "텍스처 보정 40% 확률 재시도 → 평균 2.1회"
            ),
        },
        "scenarios": {
            "A_full_nanobanana": result_a.to_dict(),
            "B_local_plus_refine": result_b.to_dict(),
            "C_local_only": result_c.to_dict(),
        },
        "comparison_A_vs_B": {
            "api_call_reduction": api_call_reduction,
            "api_call_reduction_pct": round(api_call_reduction_pct, 1),
            "cost_saving_usd": round(saving_abs, 4),
            "cost_saving_pct": round(saving_pct, 1),
        },
    }

    return report


def run_live_benchmark(pet_image: str, cloth_image: str, project_root: str) -> dict:
    """
    실제 API를 호출하여 1건 기준 비교.
    로컬 IDM-VTON 추론은 이미 완료된 상태에서 refinement만 비교.
    """
    import subprocess
    import tempfile

    project = Path(project_root).expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="petvton_bench_") as tmpdir:
        tmpdir = Path(tmpdir)

        # --- Scenario A: Full NanoBanana ---
        usage_a = NanoBananaUsage(mode="full_nanobanana")
        output_a = tmpdir / "result_full_nanobanana.jpg"

        print("[Benchmark] Scenario A: Full NanoBanana try-on...")
        t0 = time.time()
        full_tryon_nanobanana(
            pet_image_path=pet_image,
            cloth_image_path=cloth_image,
            output_path=str(output_a),
            usage=usage_a,
        )
        time_a = time.time() - t0

        # --- Scenario B: Local pipeline + NanoBanana refine ---
        # Run the local pipeline first
        local_output = tmpdir / "result_local.jpg"
        print("[Benchmark] Scenario B: Running local pipeline...")
        t1 = time.time()
        subprocess.run(
            [
                sys.executable, "-m", "backend_worker.run_pet_tryon",
                "--pet_image", pet_image,
                "--cloth_image", cloth_image,
                "--output", str(local_output),
                "--project_root", str(project),
            ],
            cwd=str(project),
            check=True,
        )
        time_local = time.time() - t1

        usage_b = NanoBananaUsage(mode="local_pipeline_plus_refine")
        output_b = tmpdir / "result_refined.jpg"

        print("[Benchmark] Scenario B: NanoBanana refinement...")
        t2 = time.time()
        refine_with_nanobanana(
            idm_vton_result_path=str(local_output),
            pet_image_path=pet_image,
            cloth_image_path=cloth_image,
            output_path=str(output_b),
            usage=usage_b,
        )
        time_refine = time.time() - t2

        report = {
            "benchmark_type": "live",
            "scenario_A": {
                **usage_a.summary(),
                "wall_time_sec": round(time_a, 2),
            },
            "scenario_B": {
                **usage_b.summary(),
                "local_pipeline_time_sec": round(time_local, 2),
                "refine_time_sec": round(time_refine, 2),
                "total_time_sec": round(time_local + time_refine, 2),
            },
            "comparison": {
                "api_call_reduction": usage_a.api_calls - usage_b.api_calls,
                "cost_saving_usd": round(
                    usage_a.estimated_cost_usd - usage_b.estimated_cost_usd, 4
                ),
            },
        }

        return report


def print_report(report: dict):
    print("\n" + "=" * 70)
    print("  Pet-VTON 비용 검증 리포트")
    print("=" * 70)

    if report["benchmark_type"] == "simulation":
        n = report["num_images"]
        print(f"\n  대상 이미지 수: {n}장")
        print(f"\n  가정:")
        assumptions = report["assumptions"]
        print(f"    - NanoBanana 이미지당 비용: ${assumptions['nanobanana_cost_per_image_edit']}")
        print(f"    - 순수 NanoBanana 평균 API 호출/건: {assumptions['avg_full_nanobanana_calls_per_tryon']}회")
        print(f"    - 로컬+리파인 평균 API 호출/건: {assumptions['avg_local_pipeline_calls_per_tryon']}회")
        print(f"    - 로컬 GPU 추론 비용/건: ${assumptions['local_gpu_cost_per_inference']}")

        for key in ["A_full_nanobanana", "B_local_plus_refine", "C_local_only"]:
            s = report["scenarios"][key]
            print(f"\n  [{s['scenario']}]")
            print(f"    API 호출 수: {s['total_api_calls']}회")
            print(f"    NanoBanana 비용: ${s['total_nanobanana_cost']:.4f}")
            print(f"    로컬 GPU 비용: ${s['local_compute_cost']:.4f}")
            print(f"    총 비용: ${s['total_cost']:.4f}")
            print(f"    이미지당 비용: ${s['cost_per_image']:.4f}")

        comp = report["comparison_A_vs_B"]
        print(f"\n  [A vs B 비교]")
        print(f"    API 호출 절감: {comp['api_call_reduction']}회 ({comp['api_call_reduction_pct']}%)")
        print(f"    비용 절감: ${comp['cost_saving_usd']:.4f} ({comp['cost_saving_pct']}%)")

        # 품질 게이트 설명
        qg = assumptions.get("quality_gate_distribution", {})
        if qg:
            print(f"\n  [품질 게이트 분포 (로컬 파이프라인)]")
            print(f"    SKIP  {qg['SKIP_ratio']:.0%}: {qg['SKIP_description']}")
            print(f"    REFINE {qg['REFINE_ratio']:.0%}: {qg['REFINE_description']}")
            print(f"    REJECT {qg['REJECT_ratio']:.0%}: {qg['REJECT_description']}")

        print(f"\n  [비용 절감 근거]")
        print(f"    A) 나노바나나만: 마스크 없이 범용 프롬프트 → 평균 {AVG_FULL_NANOBANANA_CALLS_PER_IMAGE}회 호출")
        print(f"    B) 로컬+게이트: 마스크/포즈 사전 생성 → 평균 {AVG_LOCAL_PIPELINE_CALLS_PER_IMAGE}회 호출")
        print(f"       - 50% 이미지는 IDM-VTON만으로 충분 (SKIP)")
        print(f"       - 10% 입력 불량은 사전 차단 (REJECT)")
        print(f"       - 40%만 NanoBanana 1회 호출 (마스크 기반 정밀 프롬프트)")

        if comp["cost_saving_pct"] > 0:
            print(f"\n  결론: 로컬 파이프라인 + 품질 게이트로 NanoBanana API 비용을 "
                  f"{comp['cost_saving_pct']}% 절감할 수 있습니다.")
        print()

    else:
        print("\n  [실제 API 호출 결과]")
        sa = report["scenario_A"]
        sb = report["scenario_B"]
        print(f"\n  Scenario A (순수 NanoBanana):")
        print(f"    API 호출: {sa['api_calls']}회")
        print(f"    비용: ${sa['estimated_cost_usd']:.4f}")
        print(f"    소요 시간: {sa['wall_time_sec']}s")

        print(f"\n  Scenario B (로컬 + NanoBanana 리파인):")
        print(f"    API 호출: {sb['api_calls']}회")
        print(f"    비용: ${sb['estimated_cost_usd']:.4f}")
        print(f"    로컬 파이프라인: {sb['local_pipeline_time_sec']}s")
        print(f"    NanoBanana 리파인: {sb['refine_time_sec']}s")

        comp = report["comparison"]
        print(f"\n  API 호출 절감: {comp['api_call_reduction']}회")
        print(f"  비용 절감: ${comp['cost_saving_usd']:.4f}")
        print()

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Pet-VTON 비용 검증 벤치마크")

    parser.add_argument("--mode", choices=["simulate", "live"], default="simulate",
                        help="simulate: API 호출 없이 비용 시뮬레이션 / live: 실제 API 호출 비교")
    parser.add_argument("--num_images", type=int, default=100,
                        help="시뮬레이션 대상 이미지 수 (simulate 모드)")
    parser.add_argument("--pet_image", default=None, help="live 모드용 펫 이미지 경로")
    parser.add_argument("--cloth_image", default=None, help="live 모드용 옷 이미지 경로")
    parser.add_argument("--project_root", default=str(Path.home() / "projects" / "pet-vton"))
    parser.add_argument("--output_json", default=None, help="결과를 JSON 파일로 저장")

    args = parser.parse_args()

    if args.mode == "simulate":
        report = run_simulation(args.num_images)
    else:
        if not args.pet_image or not args.cloth_image:
            parser.error("live 모드에서는 --pet_image와 --cloth_image가 필요합니다")
        report = run_live_benchmark(args.pet_image, args.cloth_image, args.project_root)

    print_report(report)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Report saved to: {args.output_json}")


if __name__ == "__main__":
    main()
