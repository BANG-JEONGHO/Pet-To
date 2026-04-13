from pathlib import Path

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "yolov8n.pt"
SOURCE_DIR = ROOT_DIR / "pet-images" / "images" / "cat" / "cat2"
OUTPUT_PROJECT_DIR = ROOT_DIR / "pet-images" / "yolo_outputs"
OUTPUT_RUN_NAME = "cat2"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    image_files = [
        path for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    if not image_files:
        raise FileNotFoundError(f"No image files found in: {SOURCE_DIR}")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(SOURCE_DIR),
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(OUTPUT_PROJECT_DIR),
        name=OUTPUT_RUN_NAME,
        exist_ok=True,
    )

    detected_images = sum(1 for result in results if len(result.boxes) > 0)
    output_dir = OUTPUT_PROJECT_DIR / OUTPUT_RUN_NAME
    label_dir = output_dir / "labels"

    print(f"Processed images: {len(image_files)}")
    print(f"Images with detections: {detected_images}")
    print(f"Annotated images: {output_dir}")
    print(f"YOLO labels: {label_dir}")


if __name__ == "__main__":
    main()
