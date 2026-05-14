from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--test-dir", default="test/images", help="Directory of test images")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--device", default="cpu", help="Device to use for inference (cpu or 0 for GPU)")
    args = parser.parse_args()

    model = YOLO(args.model)
    
    image_paths = sorted(
        [p for p in Path(args.test_dir).iterdir() if p.is_file()]
    )

    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()
        for idx, image_path in enumerate(image_paths):
            results = model.predict(source=str(image_path), conf=args.conf, save=False, verbose=False, device=args.device)
            result = results[0]
            image_id = image_path.name  # 使用真实文件名
            if result.boxes is None:
                continue
            for box in result.boxes:
                x_center, y_center, width, height = box.xywhn[0].tolist()
                writer.writerow(
                    {
                        "image_id": image_id,
                        "class_id": int(box.cls[0].item()),
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "confidence": float(box.conf[0].item()),
                    }
                )


if __name__ == "__main__":
    main()