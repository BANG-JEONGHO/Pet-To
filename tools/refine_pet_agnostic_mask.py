import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from segment_anything import SamPredictor, sam_model_registry


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def infer_pet_type(image_path: Path, pet_type: str) -> str:
    if pet_type != "auto":
        return pet_type

    name = image_path.stem.lower()

    if name.startswith("dog") or "dog-" in name or "dog_" in name:
        return "dog"

    if name.startswith("cat") or "cat-" in name or "cat_" in name:
        return "cat"

    return "unknown"


def exclusion_prompts(pet_type: str) -> List[str]:
    """
    옷 영역에서 제거할 부위.
    leg 전체를 강하게 빼면 몸통/가슴까지 날아갈 수 있으므로
    paw, face, head, ear, tail 위주로 시작한다.
    """
    if pet_type == "cat":
        return [
            "cat head",
            "cat face",
            "cat ears",
            "cat paws",
            "cat tail",
        ]

    if pet_type == "dog":
        return [
            "dog head",
            "dog face",
            "dog ears",
            "dog paws",
            "dog tail",
        ]

    return [
        "cat head",
        "cat face",
        "cat ears",
        "cat paws",
        "cat tail",
        "dog head",
        "dog face",
        "dog ears",
        "dog paws",
        "dog tail",
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
    return SamPredictor(sam)


def normalize_grounding_text(prompt: str) -> str:
    """
    Grounding DINO용 텍스트를 하나의 문자열로 정리한다.
    예: "dog head" -> "dog head."
    """
    prompt = str(prompt).strip()
    if not prompt:
        prompt = "dog"

    return prompt.rstrip(".") + "."


def detect_boxes(
    image_pil: Image.Image,
    prompt: str,
    processor,
    model,
    device: str,
    box_threshold: float,
    text_threshold: float,
):
    width, height = image_pil.size

    text_prompt = normalize_grounding_text(prompt)

    inputs = processor(
        images=image_pil,
        text=text_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

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


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return xs.min(), ys.min(), xs.max(), ys.max()


def filter_part_box(
    box: np.ndarray,
    base_mask: np.ndarray,
    max_part_box_ratio: float,
) -> bool:
    """
    GroundingDINO가 part 대신 전체 동물을 잡는 경우가 있다.
    너무 큰 box는 제거한다.
    """
    base_area = float((base_mask > 0).sum())

    if base_area <= 0:
        return False

    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    if area <= 0:
        return False

    if area > base_area * max_part_box_ratio:
        return False

    return True


def sam_part_mask_from_box(
    predictor: SamPredictor,
    box_xyxy: np.ndarray,
    base_mask: np.ndarray,
    max_part_mask_ratio: float,
) -> np.ndarray:
    """
    SAM이 part box에서도 전체 동물을 선택하는 경우가 있어
    base mask 대비 너무 큰 part mask는 버린다.
    """
    masks, scores, _ = predictor.predict(
        box=box_xyxy.astype(np.float32),
        multimask_output=True,
    )

    base_area = float((base_mask > 0).sum())
    valid = []

    for i, m in enumerate(masks):
        m = m.astype(np.uint8)
        area = float((m > 0).sum())

        if base_area > 0 and area > base_area * max_part_mask_ratio:
            continue

        valid.append((float(scores[i]), m))

    if not valid:
        return np.zeros_like(base_mask, dtype=np.uint8)

    valid.sort(key=lambda x: x[0], reverse=True)
    return valid[0][1].astype(np.uint8)


def build_exclusion_mask(
    image_pil: Image.Image,
    image_rgb: np.ndarray,
    base_mask: np.ndarray,
    pet_type: str,
    processor,
    gdino_model,
    sam_predictor,
    device: str,
    args,
) -> np.ndarray:
    sam_predictor.set_image(image_rgb)

    exclude = np.zeros_like(base_mask, dtype=np.uint8)
    prompts = exclusion_prompts(pet_type)

    for prompt in prompts:
        boxes, scores, labels = detect_boxes(
            image_pil=image_pil,
            prompt=prompt,
            processor=processor,
            model=gdino_model,
            device=device,
            box_threshold=args.part_box_threshold,
            text_threshold=args.part_text_threshold,
        )

        if len(boxes) == 0:
            continue

        for box in boxes:
            box_np = box.detach().cpu().numpy().astype(np.float32)

            if not filter_part_box(
                box=box_np,
                base_mask=base_mask,
                max_part_box_ratio=args.max_part_box_ratio,
            ):
                continue

            part_mask = sam_part_mask_from_box(
                predictor=sam_predictor,
                box_xyxy=box_np,
                base_mask=base_mask,
                max_part_mask_ratio=args.max_part_mask_ratio,
            )

            exclude = np.logical_or(exclude > 0, part_mask > 0).astype(np.uint8)

    if args.exclude_dilate_kernel > 0:
        kernel = np.ones(
            (args.exclude_dilate_kernel, args.exclude_dilate_kernel),
            dtype=np.uint8,
        )
        exclude = cv2.dilate(exclude, kernel, iterations=args.exclude_dilate_iter)

    # base_mask 바깥쪽 exclusion은 의미 없으므로 제거
    exclude = np.logical_and(exclude > 0, base_mask > 0).astype(np.uint8)

    return exclude


def geometry_torso_crop(
    mask: np.ndarray,
    top_cut_ratio: float,
    bottom_cut_ratio: float,
    side_cut_ratio: float,
) -> np.ndarray:
    """
    고양이/강아지 전체 마스크에서 상단/하단/좌우 일부를 잘라
    몸통 중심 영역으로 줄인다.

    주의:
    이 함수만으로 머리가 완전히 제거되지는 않는다.
    그래서 head/face/ear subtraction과 함께 사용한다.
    """
    bbox = mask_bbox(mask)

    if bbox is None:
        return mask

    x1, y1, x2, y2 = bbox

    w = x2 - x1 + 1
    h = y2 - y1 + 1

    nx1 = x1 + int(w * side_cut_ratio)
    nx2 = x2 - int(w * side_cut_ratio)
    ny1 = y1 + int(h * top_cut_ratio)
    ny2 = y2 - int(h * bottom_cut_ratio)

    cropped = np.zeros_like(mask, dtype=np.uint8)
    cropped[ny1 : ny2 + 1, nx1 : nx2 + 1] = mask[ny1 : ny2 + 1, nx1 : nx2 + 1]

    return cropped


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


def postprocess_final_mask(mask: np.ndarray, args) -> np.ndarray:
    mask = keep_largest_component(mask)

    if args.final_erode_kernel > 0:
        kernel = np.ones((args.final_erode_kernel, args.final_erode_kernel), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=args.final_erode_iter)

    if args.final_dilate_kernel > 0:
        kernel = np.ones((args.final_dilate_kernel, args.final_dilate_kernel), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=args.final_dilate_iter)

    mask = (mask > 0).astype(np.uint8)
    return mask


def save_preview(
    image_pil: Image.Image,
    mask_255: np.ndarray,
    preview_path: Path,
    output_size: Tuple[int, int],
):
    out_w, out_h = output_size

    image = image_pil.resize((out_w, out_h), Image.BICUBIC)
    image_np = np.array(image).astype(np.float32)

    mask_bool = mask_255 > 127

    overlay = image_np.copy()
    red = np.array([255, 0, 0], dtype=np.float32)

    overlay[mask_bool] = image_np[mask_bool] * 0.55 + red * 0.45

    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(preview_path, quality=95)


def process_one(
    image_path: Path,
    base_mask_path: Path,
    output_path: Path,
    preview_path: Path,
    processor,
    gdino_model,
    sam_predictor,
    device: str,
    args,
):
    image_pil = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image_pil)
    orig_w, orig_h = image_pil.size

    base_mask_pil = Image.open(base_mask_path).convert("L")
    base_mask_orig = np.array(
        base_mask_pil.resize((orig_w, orig_h), Image.NEAREST)
    )
    base_mask_orig = (base_mask_orig > 127).astype(np.uint8)

    pet_type = infer_pet_type(image_path, args.pet_type)

    exclude_mask = build_exclusion_mask(
        image_pil=image_pil,
        image_rgb=image_rgb,
        base_mask=base_mask_orig,
        pet_type=pet_type,
        processor=processor,
        gdino_model=gdino_model,
        sam_predictor=sam_predictor,
        device=device,
        args=args,
    )

    refined = np.logical_and(base_mask_orig > 0, exclude_mask == 0).astype(np.uint8)

    refined = geometry_torso_crop(
        refined,
        top_cut_ratio=args.top_cut_ratio,
        bottom_cut_ratio=args.bottom_cut_ratio,
        side_cut_ratio=args.side_cut_ratio,
    )

    refined = postprocess_final_mask(refined, args)

    out_w, out_h = args.width, args.height

    refined_255 = cv2.resize(
        refined.astype(np.uint8) * 255,
        (out_w, out_h),
        interpolation=cv2.INTER_NEAREST,
    )

    Image.fromarray(refined_255).save(output_path)

    save_preview(
        image_pil=image_pil,
        mask_255=refined_255,
        preview_path=preview_path,
        output_size=(out_w, out_h),
    )

    base_ratio = float((base_mask_orig > 0).mean())
    refined_ratio = float((refined > 0).mean())

    print(
        f"[OK] {image_path.name} | pet={pet_type} "
        f"| base_ratio={base_ratio:.3f} | refined_ratio={refined_ratio:.3f}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--base_mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--preview_dir", required=True)

    parser.add_argument("--pet_type", default="auto", choices=["auto", "dog", "cat"])

    parser.add_argument("--gdino_model_id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam_model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])

    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)

    # part detection threshold
    parser.add_argument("--part_box_threshold", type=float, default=0.20)
    parser.add_argument("--part_text_threshold", type=float, default=0.20)

    # too-large part filtering
    parser.add_argument("--max_part_box_ratio", type=float, default=0.45)
    parser.add_argument("--max_part_mask_ratio", type=float, default=0.40)

    # exclusion mask expansion
    parser.add_argument("--exclude_dilate_kernel", type=int, default=15)
    parser.add_argument("--exclude_dilate_iter", type=int, default=1)

    # torso geometric crop
    parser.add_argument("--top_cut_ratio", type=float, default=0.02)
    parser.add_argument("--bottom_cut_ratio", type=float, default=0.28)
    parser.add_argument("--side_cut_ratio", type=float, default=0.02)

    # final smoothing
    parser.add_argument("--final_erode_kernel", type=int, default=0)
    parser.add_argument("--final_erode_iter", type=int, default=1)
    parser.add_argument("--final_dilate_kernel", type=int, default=5)
    parser.add_argument("--final_dilate_iter", type=int, default=1)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    base_mask_dir = Path(args.base_mask_dir)
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
        stem = image_path.stem

        candidates = [
            base_mask_dir / f"{stem}.png",
            base_mask_dir / f"{stem}_mask.png",
        ]

        base_mask_path = None
        for c in candidates:
            if c.exists():
                base_mask_path = c
                break

        if base_mask_path is None:
            print(f"[WARN] no base mask: {stem}")
            continue

        output_path = output_dir / f"{stem}.png"
        preview_path = preview_dir / f"{stem}_preview.jpg"

        process_one(
            image_path=image_path,
            base_mask_path=base_mask_path,
            output_path=output_path,
            preview_path=preview_path,
            processor=processor,
            gdino_model=gdino_model,
            sam_predictor=sam_predictor,
            device=device,
            args=args,
        )

    print("done")
    print("output:", output_dir)
    print("preview:", preview_dir)


if __name__ == "__main__":
    main()
