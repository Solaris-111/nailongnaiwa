#!/usr/bin/env python3
"""用四分类模型对抽帧图做推理，按类别 + 置信度分桶（移动，不删除）。

用法:
    python tools/classify_frames.py nailong_naiwa_10_demo/naidan_frames
    python tools/classify_frames.py nailong_naiwa_10_demo/naidan_frames --conf 0.7

结果按类别移动到 frames/_classified/<类别>/ 下，低置信度帧进 _classified/low_conf/。
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image, ImageDraw

from nailong_model import load_checkpoint, load_image, predict_transform

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def safe_move(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    i = 1
    while dst.exists():
        dst = dst_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    shutil.move(str(src), str(dst))
    return dst


def mask_corners(image: Image.Image, ratio: float = 0.18) -> Image.Image:
    image = image.copy()
    w, h = image.size
    cw, ch = int(w * ratio), int(h * ratio)
    draw = ImageDraw.Draw(image)
    for box in [(0, 0, cw, ch), (w - cw, 0, w, ch), (0, h - ch, cw, h), (w - cw, h - ch, w, h)]:
        draw.rectangle(box, fill=(0, 0, 0))
    return image


def predict_masked(
    model: torch.nn.Module,
    classes: list[str],
    image_path: Path,
    device: torch.device,
    image_size: int,
    corner_ratio: float,
) -> tuple[str, dict[str, float]]:
    image = load_image(image_path)
    image = mask_corners(image, corner_ratio)
    tensor = predict_transform(image_size)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu()
    scores = {classes[i]: float(probs[i]) for i in range(len(classes))}
    predicted = max(scores, key=scores.get)
    return predicted, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="抽帧图分类分桶（移动版）")
    parser.add_argument("frames_dir", help="抽帧图片目录")
    parser.add_argument("--checkpoint", default="models/four_class_resnet18.pt",
                        help="四分类模型权重路径")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值，低于此值归 low_conf（默认 0.5）")
    parser.add_argument("--corner-ratio", type=float, default=0.18,
                        help="分类前遮挡四角的比例（默认 0.18，头像/logo 常驻区域）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, image_size, _ = load_checkpoint(args.checkpoint, device)
    print(f"模型类别: {classes}")

    frames = Path(args.frames_dir)
    images = sorted(p for p in frames.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    buckets = {c: frames / "_classified" / c for c in classes}
    low_conf_dir = frames / "_classified" / "low_conf"

    counter = Counter()
    for p in images:
        pred, scores = predict_masked(model, classes, p, device, image_size, args.corner_ratio)
        conf = scores[pred]
        dst_dir = buckets[pred] if conf >= args.conf else low_conf_dir
        safe_move(p, dst_dir)
        counter[pred if conf >= args.conf else "low_conf"] += 1
        print(f"  {p.name:45s} -> {pred:10s} {conf:.3f}")

    print("\n=== 分类汇总 ===")
    for name in classes + ["low_conf"]:
        if counter[name]:
            print(f"  {name:12s}: {counter[name]} 张")
    print(f"\n已分桶到 {frames / '_classified'}/，复核后自行归入正式数据集。")


if __name__ == "__main__":
    main()
