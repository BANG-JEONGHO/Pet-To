import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from segment_anything import SamPredictor, sam_model_registry


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def infer_pet_type(image_path: Path, pet_type: str) -> str:
    """
    파일명 기준으로 dog/cat을 자동 추정한다.
    dog-001.jpg, cat-001.jpg 형식을 지원한다.
    """
    if pet_type != "auto":
        return pet_type

    name = image_path.stem.lower()

    if name.startswith("dog") or "dog-" in name or "dog_" in name:
        return "dog"

    if name.startswith("cat") or "cat-" in name or "cat_" in name:
        return "cat"

    return "unknown"


def build_prompts(pet_type: str) -> List[str]:
    """
    Grounding DINO에 넣을 텍스트 프롬프트 후보.
    너무 구체적인 torso만 쓰면 실패할 수 있으므로 body/dog/cat fallback을 같이 둔다.
    """
    if pet_type == "dog":
        return [
            "dog body",
            "dog torso",
            "dog wearing clothes area",
            "dog",
        ]

    if pet_type == "cat":
        return [
            "cat body",
            "cat torso",
            "cat wearing clothes area",
            "cat",
        ]

    return [
        "dog body",
        "cat body",
        "dog",
        "cat",
    ]


def load_grounding_dino(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model


def load_sam(model_type: str, checkpoint: str, device: str):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor


def prompts_to_grounding_text(prompts):
    """
    transformers==4.40.2 Grounding DINO는 여러 label을
    'dog body. dog torso. dog.' 같은 하나의 문자열로 넣는 방식이 안정적이다.
    """
    cleaned = []
    for p in prompts:
        p = str(p).strip()
        if not p:
            continue
        cleaned.append(p.rstrip("."))

    if not cleaned:
        cleaned = ["dog", "cat"]

    return ". ".join(cleaned) + "."


def detect_boxes(
    image_pil: Image.Image,
    prompts: List[str],
    processor,
    model,
    device: str,
    box_threshold: float,
    text_threshold: float,
):
    """
    Grounding DINO로 텍스트 프롬프트에 해당하는 box를 찾는다.

    핵심 수정:
    - 기존: text=[["dog body", "dog torso", ...]]
    - 수정: text="dog body. dog torso. dog."

    transformers 4.40.2 기준으로 이 방식이 안정적이다.
    """
    width, height = image_pil.size

    text_prompt = prompts_to_grounding_text(prompts)

    inputs = processor(
        images=image_pil,
        text=text_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # transformers 4.40.2는 box_threshold를 사용한다.
    # 다른 버전 호환을 위해 threshold fallback도 둔다.
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(height, width)],
        )
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(height, width)],
        )

    result = results[0]

    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("labels", result.get("text_labels", []))

    return boxes, scores, labels


def choose_best_box(
    boxes,
    scores,
    labels,
    image_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    여러 box가 잡히면 가장 적절한 box 하나를 고른다.
    - score가 높을수록 좋다.
    - 면적이 너무 작은 box는 제외한다.
    - body/torso label이면 약간 가산점을 준다.
    """
    if len(boxes) == 0:
        return None

    width, height = image_size
    image_area = width * height

    best_idx = None
    best_value = -1.0

    for i, box in enumerate(boxes):
        box_np = box.detach().cpu().numpy().astype(np.float32)
        x1, y1, x2, y2 = box_np

        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        area = box_w * box_h

        if area < image_area * 0.01:
            continue

        score = float(scores[i].detach().cpu().item()) if hasattr(scores[i], "detach") else float(scores[i])

        label = str(labels[i]).lower() if i < len(labels) else ""
        keyword_bonus = 1.0
        if "body" in label or "torso" in label:
            keyword_bonus = 1.25

        # score와 면적을 같이 본다.
        value = score * keyword_bonus * np.sqrt(area / image_area)

        if value > best_value:
            best_value = value
            best_idx = i

    if best_idx is None:
        return None

    box = boxes[best_idx].detach().cpu().numpy().astype(np.float32)
    return box


def sam_mask_from_box(
    image_rgb: np.ndarray,
    box_xyxy: np.ndarray,
    predictor: SamPredictor,
) -> np.ndarray:
    """
    SAM에 box prompt를 넣어 마스크를 생성한다.
    multimask_output=True로 여러 후보를 받고, score가 가장 높은 mask를 선택한다.
    """
    predictor.set_image(image_rgb)

    masks, scores, _ = predictor.predict(
        box=box_xyxy,
        multimask_output=True,
    )

    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(np.uint8)

    return mask


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    작은 잡음 영역을 제거하고 가장 큰 연결 영역만 남긴다.
    """
    mask_uint8 = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    if num_labels <= 1:
        return mask_uint8

    # 0번은 배경
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest = (labels == largest_label).astype(np.uint8)

    return largest


def soft_torso_refine(
    mask: np.ndarray,
    top_cut_ratio: float,
    bottom_cut_ratio: float,
    side_cut_ratio: float,
) -> np.ndarray:
    """
    간단한 torso 후처리.
    주의:
    동물의 방향이 다양하기 때문에 완벽한 torso parsing은 아니다.
    처음에는 full mode로 확인하고, 너무 넓게 잡히면 soft_torso를 사용한다.
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return mask

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    w = x2 - x1 + 1
    h = y2 - y1 + 1

    nx1 = x1 + int(w * side_cut_ratio)
    nx2 = x2 - int(w * side_cut_ratio)
    ny1 = y1 + int(h * top_cut_ratio)
    ny2 = y2 - int(h * bottom_cut_ratio)

    refined = np.zeros_like(mask, dtype=np.uint8)
    refined[ny1 : ny2 + 1, nx1 : nx2 + 1] = mask[ny1 : ny2 + 1, nx1 : nx2 + 1]

    return refined


def postprocess_mask(
    mask: np.ndarray,
    output_size: Tuple[int, int],
    mask_mode: str,
    dilate_kernel: int,
    dilate_iter: int,
    top_cut_ratio: float,
    bottom_cut_ratio: float,
    side_cut_ratio: float,
) -> np.ndarray:
    """
    최종 IDM-VTON용 binary mask 생성.
    output_size는 (width, height).
    """
    mask = keep_largest_component(mask)

    if mask_mode == "soft_torso":
        mask = soft_torso_refine(
            mask,
            top_cut_ratio=top_cut_ratio,
            bottom_cut_ratio=bottom_cut_ratio,
            side_cut_ratio=side_cut_ratio,
        )
        mask = keep_largest_component(mask)

    if dilate_kernel > 0 and dilate_iter > 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilate_iter)

    out_w, out_h = output_size
    mask = cv2.resize(mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255

    return mask


def save_preview(
    image_pil: Image.Image,
    mask_255: np.ndarray,
    preview_path: Path,
    output_size: Tuple[int, int],
):
    """
    마스크가 어디 잡혔는지 확인하기 위한 preview 이미지 저장.
    """
    out_w, out_h = output_size

    image = image_pil.resize((out_w, out_h), Image.BICUBIC)
    image_np = np.array(image).astype(np.float32)

    mask_bool = mask_255 > 127

    overlay = image_np.copy()
    color = np.array([255, 0, 0], dtype=np.float32)

    overlay[mask_bool] = image_np[mask_bool] * 0.55 + color * 0.45

    preview = overlay.clip(0, 255).astype(np.uint8)
    Image.fromarray(preview).save(preview_path, quality=95)


def process_one_image(
    image_path: Path,
    output_dir: Path,
    preview_dir: Path,
    processor,
    gdino_model,
    sam_predictor,
    device: str,
    args,
):
    image_pil = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image_pil)

    current_pet_type = infer_pet_type(image_path, args.pet_type)
    prompts = build_prompts(current_pet_type)

    boxes, scores, labels = detect_boxes(
        image_pil=image_pil,
        prompts=prompts,
        processor=processor,
        model=gdino_model,
        device=device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    box = choose_best_box(
        boxes=boxes,
        scores=scores,
        labels=labels,
        image_size=image_pil.size,
    )

    # box가 안 잡히면 threshold를 낮춰서 한 번 더 시도
    if box is None:
        boxes, scores, labels = detect_boxes(
            image_pil=image_pil,
            prompts=prompts,
            processor=processor,
            model=gdino_model,
            device=device,
            box_threshold=max(0.15, args.box_threshold - 0.10),
            text_threshold=max(0.15, args.text_threshold - 0.10),
        )

        box = choose_best_box(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_size=image_pil.size,
        )

    if box is None:
        print(f"[WARN] no box detected: {image_path.name}")
        empty = np.zeros((args.height, args.width), dtype=np.uint8)

        Image.fromarray(empty).save(output_dir / f"{image_path.stem}.png")
        Image.fromarray(empty).save(output_dir / f"{image_path.stem}_mask.png")

        return

    raw_mask = sam_mask_from_box(
        image_rgb=image_rgb,
        box_xyxy=box,
        predictor=sam_predictor,
    )

    mask_255 = postprocess_mask(
        mask=raw_mask,
        output_size=(args.width, args.height),
        mask_mode=args.mask_mode,
        dilate_kernel=args.dilate_kernel,
        dilate_iter=args.dilate_iter,
        top_cut_ratio=args.top_cut_ratio,
        bottom_cut_ratio=args.bottom_cut_ratio,
        side_cut_ratio=args.side_cut_ratio,
    )

    # 두 가지 이름으로 저장:
    # 1) dog-001.png
    # 2) dog-001_mask.png
    # IDM-VTON loader가 어떤 naming을 쓰는지 확인 전까지 둘 다 두면 안전하다.
    Image.fromarray(mask_255).save(output_dir / f"{image_path.stem}.png")
    Image.fromarray(mask_255).save(output_dir / f"{image_path.stem}_mask.png")

    save_preview(
        image_pil=image_pil,
        mask_255=mask_255,
        preview_path=preview_dir / f"{image_path.stem}_preview.jpg",
        output_size=(args.width, args.height),
    )

    ratio = (mask_255 > 127).mean()
    print(f"[OK] {image_path.name} | pet={current_pet_type} | mask_ratio={ratio:.3f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--preview_dir", required=True)

    parser.add_argument("--pet_type", default="auto", choices=["auto", "dog", "cat"])

    parser.add_argument("--gdino_model_id", default="IDEA-Research/grounding-dino-tiny")

    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam_model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])

    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)

    parser.add_argument("--box_threshold", type=float, default=0.30)
    parser.add_argument("--text_threshold", type=float, default=0.25)

    parser.add_argument("--mask_mode", default="full", choices=["full", "soft_torso"])

    parser.add_argument("--dilate_kernel", type=int, default=21)
    parser.add_argument("--dilate_iter", type=int, default=1)

    parser.add_argument("--top_cut_ratio", type=float, default=0.08)
    parser.add_argument("--bottom_cut_ratio", type=float, default=0.12)
    parser.add_argument("--side_cut_ratio", type=float, default=0.03)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    preview_dir = Path(args.preview_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    print("[1/2] loading Grounding DINO...")
    processor, gdino_model = load_grounding_dino(
        model_id=args.gdino_model_id,
        device=device,
    )

    print("[2/2] loading SAM...")
    sam_predictor = load_sam(
        model_type=args.sam_model_type,
        checkpoint=args.sam_checkpoint,
        device=device,
    )

    image_paths = [
        p for p in sorted(input_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTS
    ]

    print("num images:", len(image_paths))

    for image_path in tqdm(image_paths):
        process_one_image(
            image_path=image_path,
            output_dir=output_dir,
            preview_dir=preview_dir,
            processor=processor,
            gdino_model=gdino_model,
            sam_predictor=sam_predictor,
            device=device,
            args=args,
        )

    print("done")
    print("mask dir:", output_dir)
    print("preview dir:", preview_dir)


if __name__ == "__main__":
    main()
