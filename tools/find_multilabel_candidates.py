#!/usr/bin/env python3
"""用四分类模型找同框候选：奶龙/奶蛙/奶蛋中至少两类概率都高于阈值。"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nailong_model import load_checkpoint, predict_image

POSITIVE = ["nailong", "naiwa", "naidan"]


def main() -> None:
    parser = argparse.ArgumentParser(description="找同框候选")
    parser.add_argument("frames_dir")
    parser.add_argument("--checkpoint", default="models/four_class_resnet18_gpu.pt")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="正类概率阈值，至少两类超过才算同框候选")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, image_size, _ = load_checkpoint(args.checkpoint, device)

    frames = sorted(p for p in Path(args.frames_dir).glob("*.jpg"))
    candidates = []
    for p in frames:
        _, scores = predict_image(model, classes, p, device, image_size)
        pos = {c: scores[c] for c in POSITIVE}
        high = [c for c in POSITIVE if pos[c] > args.threshold]
        if len(high) >= 2:
            candidates.append((p.name, pos))

    for name, pos in candidates:
        s = "  ".join(f"{c}={pos[c]:.2f}" for c in POSITIVE)
        print(f"{name}: {s}")
    print(f"\n共 {len(candidates)} 个同框候选（阈值 {args.threshold}）")


if __name__ == "__main__":
    main()
