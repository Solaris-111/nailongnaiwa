#!/usr/bin/env python3
"""用多标签模型筛同框候选，连续帧去重后复制到一个目录供人工挑。"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nailong_model import load_checkpoint, load_image, predict_transform


def main() -> None:
    parser = argparse.ArgumentParser(description="筛同框候选 + 去重 + 复制")
    parser.add_argument("frames_dir")
    parser.add_argument("--checkpoint", default="models/multilabel_resnet18.pt")
    parser.add_argument("--out", default="nailong_naiwa_10_demo/cooccur_candidates")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--gap", type=int, default=4, help="连续帧去重间隔（帧号差小于此值视为连续）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, image_size, _ = load_checkpoint(args.checkpoint, device)
    idx = {name: i for i, name in enumerate(classes)}

    frames = sorted(Path(args.frames_dir).glob("*.jpg"))
    candidates = []
    for p in frames:
        image = load_image(p)
        tensor = predict_transform(image_size)(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu()
        if float(probs[idx["naiwa"]]) > args.threshold and float(probs[idx["naidan"]]) > args.threshold:
            candidates.append(p)

    candidates.sort(key=lambda p: p.name)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    picked = []
    last_key = None
    last_num = -999
    for p in candidates:
        stem = p.stem
        key = stem.rsplit("_f", 1)[0]
        num = int(stem.rsplit("_f", 1)[1])
        if key == last_key and abs(num - last_num) < args.gap:
            continue
        picked.append(p)
        last_key = key
        last_num = num

    for p in picked:
        shutil.copy2(p, out_dir / p.name)

    print(f"候选 {len(candidates)} 张 -> 去重后 {len(picked)} 张 -> {out_dir}")


if __name__ == "__main__":
    main()
