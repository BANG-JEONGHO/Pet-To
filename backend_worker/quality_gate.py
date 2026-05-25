"""
품질 게이트: 로컬 전처리 결과를 분석하여 NanoBanana API 호출 여부를 결정.

판정 결과:
  - SKIP:   IDM-VTON 결과가 충분히 좋음 → NanoBanana 호출 불필요 (비용 $0)
  - REFINE: 약간의 보정 필요 → NanoBanana 1회 호출 (비용 $0.02)
  - REJECT: 입력 자체가 불량 → API 호출 차단 (헛돈 방지)

판단 기준:
  1. 마스크 커버리지: agnostic mask가 이미지의 적정 비율을 차지하는지
  2. 마스크 연결성: 마스크가 하나의 연결 영역인지 (파편화 여부)
  3. 의류-마스크 정합성: IDM-VTON 결과에서 마스크 영역의 색상 변화 정도
  4. 아티팩트 검출: IDM-VTON 결과의 경계 부분 부자연스러움 측정
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


class GateDecision(str, Enum):
    SKIP = "SKIP"       # NanoBanana 불필요 → API 호출 0회
    REFINE = "REFINE"   # NanoBanana 후처리 필요 → API 호출 1회
    REJECT = "REJECT"   # 입력 불량 → API 호출 차단


@dataclass
class QualityReport:
    decision: GateDecision
    score: float               # 0.0 ~ 1.0 (높을수록 품질 좋음)
    mask_coverage: float       # 마스크가 이미지에서 차지하는 비율
    mask_components: int       # 마스크 연결 영역 수
    boundary_smoothness: float # 경계 부드러움 (높을수록 좋음)
    color_consistency: float   # 의류 영역 색상 일관성 (높을수록 좋음)
    reasons: list              # 판단 근거

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "score": round(self.score, 4),
            "mask_coverage": round(self.mask_coverage, 4),
            "mask_components": self.mask_components,
            "boundary_smoothness": round(self.boundary_smoothness, 4),
            "color_consistency": round(self.color_consistency, 4),
            "reasons": self.reasons,
        }


# ──────────────────────────────────────────────
#  개별 품질 지표 계산
# ──────────────────────────────────────────────

def calc_mask_coverage(mask: np.ndarray) -> float:
    """마스크가 이미지에서 차지하는 비율 (0~1)."""
    total = mask.shape[0] * mask.shape[1]
    if total == 0:
        return 0.0
    return float((mask > 127).sum()) / total


def calc_mask_components(mask: np.ndarray) -> int:
    """마스크의 연결 영역 수. 1이면 깨끗, 많으면 파편화."""
    binary = (mask > 127).astype(np.uint8)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return max(0, num_labels - 1)  # 배경 제외


def calc_boundary_smoothness(
    result_img: np.ndarray,
    mask: np.ndarray,
    kernel_size: int = 5,
) -> float:
    """
    IDM-VTON 결과에서 마스크 경계의 부드러움 측정.
    경계 근처 픽셀의 gradient 크기가 작을수록 자연스러움.
    반환: 0~1 (1이면 매우 부드러움).
    """
    binary = (mask > 127).astype(np.uint8)

    # 마스크 경계 추출 (dilate - erode)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    boundary = ((dilated - eroded) > 0)

    if boundary.sum() == 0:
        return 1.0

    # 결과 이미지의 gradient 크기
    gray = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 경계 영역의 평균 gradient
    boundary_grad = float(grad_mag[boundary].mean())

    # 정규화: gradient가 작을수록 점수 높음
    # 경험적으로 gradient 50 이하면 매우 부드러움, 200 이상이면 거칠음
    smoothness = max(0.0, 1.0 - boundary_grad / 200.0)
    return min(1.0, smoothness)


def calc_color_consistency(
    result_img: np.ndarray,
    cloth_img: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    IDM-VTON 결과의 마스크 영역 색상이 원본 의류와 얼마나 일치하는지.
    히스토그램 비교 사용. 반환: 0~1 (1이면 완벽 일치).
    """
    binary = (mask > 127)

    if binary.sum() == 0:
        return 0.0

    # 결과에서 마스크 영역의 색상 히스토그램
    result_hsv = cv2.cvtColor(result_img, cv2.COLOR_RGB2HSV)
    cloth_hsv = cv2.cvtColor(cloth_img, cv2.COLOR_RGB2HSV)

    mask_uint8 = binary.astype(np.uint8) * 255

    # H, S 채널 기준 2D 히스토그램 비교
    h_bins, s_bins = 30, 32
    hist_range = [0, 180, 0, 256]

    hist_result = cv2.calcHist(
        [result_hsv], [0, 1], mask_uint8,
        [h_bins, s_bins], hist_range
    )
    cv2.normalize(hist_result, hist_result)

    hist_cloth = cv2.calcHist(
        [cloth_hsv], [0, 1], None,
        [h_bins, s_bins], hist_range
    )
    cv2.normalize(hist_cloth, hist_cloth)

    # 상관계수 기반 비교 (1이면 동일 분포)
    similarity = cv2.compareHist(hist_result, hist_cloth, cv2.HISTCMP_CORREL)
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))  # -1~1 → 0~1


# ──────────────────────────────────────────────
#  품질 게이트 메인 함수
# ──────────────────────────────────────────────

# 임계값 설정
MASK_COVERAGE_MIN = 0.03       # 최소 3% 이상 마스크 영역 필요
MASK_COVERAGE_MAX = 0.70       # 70% 초과시 마스크 과잉 (오탐지)
MASK_COMPONENTS_MAX = 3        # 연결 영역 3개 이하
BOUNDARY_SMOOTHNESS_GOOD = 0.6 # 이 이상이면 경계 양호
COLOR_CONSISTENCY_GOOD = 0.55  # 이 이상이면 색상 정합 양호
OVERALL_SKIP_THRESHOLD = 0.70  # 종합 점수 이 이상이면 NanoBanana 스킵
OVERALL_REJECT_THRESHOLD = 0.25 # 종합 점수 이 이하면 입력 불량


def evaluate_quality(
    result_image_path: str,
    mask_path: str,
    cloth_image_path: str,
    pet_image_path: Optional[str] = None,
) -> QualityReport:
    """
    IDM-VTON 결과를 분석하여 NanoBanana 호출 여부를 결정한다.

    Args:
        result_image_path: IDM-VTON 추론 결과 이미지
        mask_path:         agnostic mask (refined)
        cloth_image_path:  원본 의류 이미지
        pet_image_path:    원본 펫 이미지 (선택)

    Returns:
        QualityReport with decision (SKIP / REFINE / REJECT)
    """
    # 이미지 로드
    result_img = np.array(Image.open(result_image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    cloth_img = np.array(Image.open(cloth_image_path).convert("RGB"))

    # 마스크를 결과 이미지 크기에 맞춤
    h, w = result_img.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    cloth_img_resized = cv2.resize(cloth_img, (w, h), interpolation=cv2.INTER_LINEAR)

    reasons = []

    # 1. 마스크 커버리지
    coverage = calc_mask_coverage(mask)
    if coverage < MASK_COVERAGE_MIN:
        reasons.append(f"마스크 커버리지 부족 ({coverage:.1%} < {MASK_COVERAGE_MIN:.0%}): 펫 탐지 실패 가능")
    elif coverage > MASK_COVERAGE_MAX:
        reasons.append(f"마스크 커버리지 과잉 ({coverage:.1%} > {MASK_COVERAGE_MAX:.0%}): 오탐지 가능")

    # 2. 마스크 연결성
    components = calc_mask_components(mask)
    if components > MASK_COMPONENTS_MAX:
        reasons.append(f"마스크 파편화 ({components}개 영역): 탐지 불안정")

    # 3. 경계 부드러움
    smoothness = calc_boundary_smoothness(result_img, mask)
    if smoothness < BOUNDARY_SMOOTHNESS_GOOD:
        reasons.append(f"경계 부자연스러움 (smoothness={smoothness:.2f})")
    else:
        reasons.append(f"경계 양호 (smoothness={smoothness:.2f})")

    # 4. 색상 정합성
    consistency = calc_color_consistency(result_img, cloth_img_resized, mask)
    if consistency < COLOR_CONSISTENCY_GOOD:
        reasons.append(f"의류 색상 불일치 (consistency={consistency:.2f})")
    else:
        reasons.append(f"의류 색상 양호 (consistency={consistency:.2f})")

    # 종합 점수 계산 (가중 평균)
    weights = {
        "coverage": 0.20,
        "components": 0.15,
        "smoothness": 0.35,
        "consistency": 0.30,
    }

    # 커버리지 점수: 적정 범위(5~50%)에서 1.0, 범위 밖은 감소
    if MASK_COVERAGE_MIN <= coverage <= MASK_COVERAGE_MAX:
        coverage_score = 1.0
    elif coverage < MASK_COVERAGE_MIN:
        coverage_score = coverage / MASK_COVERAGE_MIN
    else:
        coverage_score = max(0.0, 1.0 - (coverage - MASK_COVERAGE_MAX) / 0.3)

    # 연결성 점수
    component_score = 1.0 if components <= 1 else max(0.0, 1.0 - (components - 1) * 0.3)

    overall = (
        weights["coverage"] * coverage_score
        + weights["components"] * component_score
        + weights["smoothness"] * smoothness
        + weights["consistency"] * consistency
    )
    overall = max(0.0, min(1.0, overall))

    # 판정
    if coverage < MASK_COVERAGE_MIN or coverage > MASK_COVERAGE_MAX:
        decision = GateDecision.REJECT
        reasons.insert(0, "REJECT: 마스크 품질 불량 → API 호출 차단")
    elif components > MASK_COMPONENTS_MAX + 2:
        decision = GateDecision.REJECT
        reasons.insert(0, "REJECT: 마스크 심각한 파편화 → API 호출 차단")
    elif overall >= OVERALL_SKIP_THRESHOLD:
        decision = GateDecision.SKIP
        reasons.insert(0, f"SKIP: 품질 충분 (score={overall:.2f}) → NanoBanana 불필요")
    else:
        decision = GateDecision.REFINE
        reasons.insert(0, f"REFINE: 보정 필요 (score={overall:.2f}) → NanoBanana 1회 호출")

    return QualityReport(
        decision=decision,
        score=overall,
        mask_coverage=coverage,
        mask_components=components,
        boundary_smoothness=smoothness,
        color_consistency=consistency,
        reasons=reasons,
    )


def evaluate_input_viability(
    mask_path: str,
) -> Tuple[bool, str]:
    """
    NanoBanana 호출 전 입력 검증.
    마스크가 유효하지 않으면 API 호출 자체를 차단한다.

    Returns:
        (viable, reason)
    """
    try:
        mask = np.array(Image.open(mask_path).convert("L"))
    except Exception as e:
        return False, f"마스크 파일 로드 실패: {e}"

    coverage = calc_mask_coverage(mask)
    if coverage < MASK_COVERAGE_MIN:
        return False, f"마스크 커버리지 {coverage:.1%}: 펫 탐지 실패로 API 호출 차단"

    if coverage > MASK_COVERAGE_MAX:
        return False, f"마스크 커버리지 {coverage:.1%}: 오탐지로 API 호출 차단"

    components = calc_mask_components(mask)
    if components > MASK_COMPONENTS_MAX + 2:
        return False, f"마스크 {components}개 파편: 탐지 불안정으로 API 호출 차단"

    return True, "입력 유효"
