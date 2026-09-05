#!/usr/bin/env python3
"""图像去重（移动版）：md5 精确重复 + pHash 相似图，只移动不删除。

用法:
    python tools/dedup_images.py nailong_naiwa_10_demo/nailong
    python tools/dedup_images.py nailong_naiwa_10_demo/nailong --threshold 5

依赖: pip install pillow imagehash
"""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image
import imagehash

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def safe_move(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    i = 1
    while dst.exists():
        dst = dst_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    src.rename(dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="图像去重（移动版）")
    parser.add_argument("src", help="要清理的图片目录")
    parser.add_argument("--threshold", type=int, default=8,
                        help="pHash 汉明距离阈值，越小越严格（默认 8，≤5 几乎同一张，≤10 同场景）")
    parser.add_argument("--out", default="_similar",
                        help="相似图移动到的子目录（默认 src/_similar）")
    args = parser.parse_args()

    src = Path(args.src)
    images = sorted(p for p in src.iterdir() if p.is_file() and is_image(p))
    print(f"共 {len(images)} 张图")

    # 1) md5 精确重复：每组保留文件最大的一张，其余移走
    md5_groups = defaultdict(list)
    for p in images:
        md5_groups[md5_of(p)].append(p)
    dup_groups = [g for g in md5_groups.values() if len(g) > 1]
    md5_dir = src / args.out / "md5_dup"
    n_md5 = 0
    for group in dup_groups:
        group.sort(key=lambda p: -p.stat().st_size)
        for p in group[1:]:
            safe_move(p, md5_dir)
            n_md5 += 1
    print(f"[md5] 精确重复 {len(dup_groups)} 组，移走 {n_md5} 张")

    # 2) pHash 相似：对 md5 去重后的剩余图，贪心比对汉明距离
    remaining = sorted(p for p in src.iterdir() if p.is_file() and is_image(p))
    phash_dir = src / args.out / "phash_similar"
    kept = []
    n_phash = 0
    for p in remaining:
        try:
            h = imagehash.phash(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"  跳过 {p.name}: {e}")
            continue
        if any(h - kh <= args.threshold for _, kh in kept):
            safe_move(p, phash_dir)
            n_phash += 1
        else:
            kept.append((p, h))
    print(f"[pHash] 阈值 {args.threshold}，移走 {n_phash} 张")

    print(f"保留 {len(kept)} 张。相似图已移到 {src / args.out}/ 下，review 后自行删除。")


if __name__ == "__main__":
    main()
