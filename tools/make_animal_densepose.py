import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from densepose import add_densepose_config


CAT_CLASS_ID = 2
DOG_CLASS_ID = 3
TARGET_CLASS_IDS = {CAT_CLASS_ID, DOG_CLASS_ID}


def build_predictor(config_file: str, weights: str, score_thresh: float):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    return DefaultPredictor(cfg)


def embedding_to_rgb(embedding_3chw: torch.Tensor) -> np.ndarray:
    """
    CSE embedding의 앞 3개 채널을 RGB 이미지처럼 변환한다.

    주의:
    이미지마다 min-max normalize를 하면 색상 기준이 계속 바뀔 수 있다.
    그래서 tanh를 사용해 값 범위를 안정화한다.
    """
    x = embedding_3chw.detach().cpu().float().numpy()
    x = np.transpose(x, (1, 2, 0))  # [H, W, 3]

    x = np.tanh(x)
    x = ((x + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return x


def render_densepose_image(image_bgr: np.ndarray, outputs, width: int, height: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    instances = outputs["instances"].to("cpu")

    if not instances.has("pred_densepose"):
        return np.array(Image.fromarray(canvas).resize((width, height), Image.BICUBIC))

    boxes = instances.pred_boxes.tensor
    scores = instances.scores
    classes = instances.pred_classes
    densepose = instances.pred_densepose

    embeddings = densepose.embedding       # [N, D, S, S]
    coarse_segm = densepose.coarse_segm    # [N, 2, S, S]

    order = torch.argsort(scores, descending=True)

    for idx in order.tolist():
        cls_id = int(classes[idx].item())

        # 강아지/고양이만 사용
        if cls_id not in TARGET_CLASS_IDS:
            continue

        x0, y0, x1, y1 = boxes[idx].round().int().tolist()

        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h))

        box_w = x1 - x0
        box_h = y1 - y0

        if box_w <= 1 or box_h <= 1:
            continue

        emb = embeddings[idx : idx + 1]
        seg = coarse_segm[idx : idx + 1]

        emb_resized = F.interpolate(
            emb,
            size=(box_h, box_w),
            mode="bilinear",
            align_corners=False,
        )[0]

        seg_resized = F.interpolate(
            seg,
            size=(box_h, box_w),
            mode="bilinear",
            align_corners=False,
        )[0]

        fg_mask = seg_resized.argmax(dim=0).numpy().astype(bool)
        rgb_patch = embedding_to_rgb(emb_resized[:3])

        target = canvas[y0:y1, x0:x1]
        target[fg_mask] = rgb_patch[fg_mask]
        canvas[y0:y1, x0:x1] = target

    canvas = np.array(Image.fromarray(canvas).resize((width, height), Image.BICUBIC))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor(
        config_file=args.config_file,
        weights=args.weights,
        score_thresh=args.score_thresh,
    )

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(input_dir.glob(ext))

    image_paths = sorted(image_paths)

    for image_path in tqdm(image_paths):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"skip unreadable image: {image_path}")
            continue

        outputs = predictor(image_bgr)

        pose_rgb = render_densepose_image(
            image_bgr=image_bgr,
            outputs=outputs,
            width=args.width,
            height=args.height,
        )

        out_path = output_dir / f"{image_path.stem}.jpg"
        Image.fromarray(pose_rgb).save(out_path, quality=95)

    print(f"done. saved to: {output_dir}")


if __name__ == "__main__":
    main()
