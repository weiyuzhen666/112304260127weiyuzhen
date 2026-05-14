import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from ultralytics import YOLO


def main():
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=50,
        batch=4,
        imgsz=640,
        device=device,
        workers=0,
        optimizer='SGD',
        lr0=0.01,
        resume=True,
        amp=False,
    )
    
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()