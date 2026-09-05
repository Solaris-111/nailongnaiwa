#!/usr/bin/env python3
"""用多标签模型筛选同框候选：奶蛙和奶蛋概率都高于阈值。"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nailong_model import load_checkpoint, load_image, predict_transform, MULTILABEL_CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(description="多标签模型筛同框候选")
    parser.add_argument("frames_dir")
    parser.add_argument("--checkpoint", default="models/multilabel_resnet18.pt")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="奶蛙和奶蛋概率阈值，两者都超过才算候选")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, image_size, _ = load_checkpoint(args.checkpoint, device)
    idx = {name: i for i, name in enumerate(classes)}

    frames = sorted(p for p in Path(args.frames_dir).glob("*.jpg"))
    candidates = []
    for p in frames:
        image = load_image(p)
        tensor = predict_transform(image_size)(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu()
        naiwa = float(probs[idx["naiwa"]])
        naidan = float(probs[idx["naidan"]])
        if naiwa > args.threshold and naidan > args.threshold:
            candidates.append((p.name, naiwa, naidan))

    for name, naiwa, naidan in candidates:
        print(f"{name}: naiwa={naiwa:.2f} naidan={naidan:.2f}")
    print(f"\n共 {len(candidates)} 个同框候选（奶蛙和奶蛋都 > {args.threshold}）")


if __name__ == "__main__":
    main()
