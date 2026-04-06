from __future__ import annotations

from pathlib import Path
import shutil


ROOT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = ROOT_DIR / "images"
LIST_PATH = ROOT_DIR / "annotations" / "list.txt"


def parse_species_map(list_path: Path) -> dict[str, str]:
    species_map: dict[str, str] = {}

    with list_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            image_id, _class_id, species_id, _breed_id = line.split()
            species_map[image_id] = "cat" if species_id == "1" else "dog"

    return species_map


def organize_images() -> tuple[int, int]:
    species_map = parse_species_map(LIST_PATH)

    cat_dir = IMAGES_DIR / "cat"
    dog_dir = IMAGES_DIR / "dog"
    cat_dir.mkdir(exist_ok=True)
    dog_dir.mkdir(exist_ok=True)

    moved = 0
    skipped = 0

    for image_path in IMAGES_DIR.glob("*.jpg"):
        target_group = species_map.get(image_path.stem)
        if target_group is None:
            skipped += 1
            continue

        target_dir = cat_dir if target_group == "cat" else dog_dir
        target_path = target_dir / image_path.name

        if target_path.exists():
            skipped += 1
            continue

        shutil.move(str(image_path), str(target_path))
        moved += 1

    return moved, skipped


def organize_remaining_by_name() -> tuple[int, int]:
    cat_dir = IMAGES_DIR / "cat"
    dog_dir = IMAGES_DIR / "dog"
    cat_dir.mkdir(exist_ok=True)
    dog_dir.mkdir(exist_ok=True)

    moved = 0
    skipped = 0

    for image_path in IMAGES_DIR.glob("*.jpg"):
        first_char = image_path.stem[:1]
        if not first_char:
            skipped += 1
            continue

        target_dir = cat_dir if first_char.isupper() else dog_dir
        target_path = target_dir / image_path.name

        if target_path.exists():
            skipped += 1
            continue

        shutil.move(str(image_path), str(target_path))
        moved += 1

    return moved, skipped


if __name__ == "__main__":
    moved_count, skipped_count = organize_images()
    fallback_moved_count, fallback_skipped_count = organize_remaining_by_name()
    print(f"Moved by list.txt: {moved_count}")
    print(f"Skipped by list.txt: {skipped_count}")
    print(f"Moved by filename rule: {fallback_moved_count}")
    print(f"Skipped by filename rule: {fallback_skipped_count}")
