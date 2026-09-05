#!/usr/bin/env python3
"""奶龙奶蛙奶蛋数据流水线：抽帧 → 去重 → 分类 → (人工挑) → 清坏图 → prepare → train。

分步用法（在项目根目录运行）:
    python tools/pipeline.py extract  视频.mp4 [更多视频...] --out nailong_naiwa_10_demo/naidan_frames
    python tools/pipeline.py dedup    nailong_naiwa_10_demo/naidan_frames --threshold 5
    python tools/pipeline.py classify nailong_naiwa_10_demo/naidan_frames
    python tools/pipeline.py clean    --base nailong_naiwa_10_demo --dirs naidan nailong naiwa unrelated
    python tools/pipeline.py prepare
    python tools/pipeline.py train

抽帧后人工挑纯类别图归入正式目录，再继续 clean → prepare → train。
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FFMPEG = shutil.which("ffmpeg") or (
    r"C:\Users\asus\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe"
)
PYTHON = sys.executable
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}


def run(script: str, *args: str) -> None:
    subprocess.run([PYTHON, str(ROOT / "tools" / script), *args], cwd=ROOT, check=True)


def extract(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos: list[Path] = []
    for item in args.videos:
        p = Path(item)
        if p.is_dir():
            videos += [f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS]
        elif p.is_file():
            videos.append(p)
    for v in videos:
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(v),
             "-vf", f"fps={args.fps}", "-q:v", "2",
             str(out / f"{v.stem}_f%03d.jpg")],
            check=True,
        )
        print(f"抽帧完成: {v.name}")
    print(f"共抽帧 {len(list(out.glob('*.jpg')))} 张 -> {out}")


def clean(args: argparse.Namespace) -> None:
    from PIL import Image
    bad_dir = Path(args.base) / "_bad"
    moved = 0
    for d in args.dirs:
        base = Path(args.base) / d
        for p in sorted(base.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                Image.open(p).convert("RGB")
            except Exception:
                bad_dir.mkdir(exist_ok=True)
                p.rename(bad_dir / f"{d}_{p.name}")
                moved += 1
    print(f"移走 {moved} 个坏图 -> {bad_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="奶龙奶蛙奶蛋数据流水线")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="视频抽帧")
    p.add_argument("videos", nargs="+", help="视频文件或目录")
    p.add_argument("--out", default="nailong_naiwa_10_demo/naidan_frames")
    p.add_argument("--fps", type=float, default=0.5, help="抽帧间隔（帧/秒）")
    p.set_defaults(func=extract)

    p = sub.add_parser("dedup", help="pHash 去重")
    p.add_argument("dir")
    p.add_argument("--threshold", type=int, default=5)
    p.set_defaults(func=lambda a: run("dedup_images.py", a.dir, "--threshold", str(a.threshold)))

    p = sub.add_parser("classify", help="四分类分桶")
    p.add_argument("dir")
    p.add_argument("--checkpoint", default="models/four_class_resnet18_gpu.pt")
    p.set_defaults(func=lambda a: run("classify_frames.py", a.dir, "--checkpoint", a.checkpoint))

    p = sub.add_parser("clean", help="清坏图")
    p.add_argument("--base", default="nailong_naiwa_10_demo")
    p.add_argument("--dirs", nargs="+", default=["naidan", "nailong", "naiwa", "unrelated"])
    p.set_defaults(func=clean)

    p = sub.add_parser("prepare", help="生成 splits")
    p.set_defaults(func=lambda a: run("prepare_four_class.py"))

    p = sub.add_parser("train", help="训练")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out", default="models/four_class_resnet18_gpu.pt")
    p.set_defaults(func=lambda a: run(
        "train_pretrained_classifier.py",
        "--split-dir", "nailong_naiwa_splits_4class",
        "--out", a.out, "--epochs", str(a.epochs),
    ))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
