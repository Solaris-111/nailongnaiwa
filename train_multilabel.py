from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from nailong_model import (
    MULTILABEL_CLASSES,
    MultiLabelDataset,
    build_torchvision_classifier,
    save_checkpoint,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SINGLE_LABELS = {
    "nailong": [1, 0, 0],
    "naiwa": [0, 1, 0],
    "naidan": [0, 0, 1],
    "unrelated": [0, 0, 0],
}


def build_items(base_dir: Path, manifest: Path | None) -> list[tuple[str, list[int]]]:
    items: list[tuple[str, list[int]]] = []
    for cls, vec in SINGLE_LABELS.items():
        d = base_dir / cls
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                items.append((str(p), vec))
    if manifest and manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for path, vec in data.items():
            items.append((str(path), [int(x) for x in vec]))
    return items


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5):
    model.eval()
    exact = 0
    total = 0
    per_class_correct = [0] * len(MULTILABEL_CLASSES)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = (torch.sigmoid(model(images)) > threshold).float()
            exact += int((preds == labels).all(dim=1).sum().item())
            total += labels.size(0)
            for c in range(labels.size(1)):
                per_class_correct[c] += int((preds[:, c] == labels[:, c]).sum().item())
    exact_match = exact / max(total, 1)
    per_class = [per_class_correct[c] / max(total, 1) for c in range(len(MULTILABEL_CLASSES))]
    return exact_match, per_class


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch_cache = Path(".torch_cache")
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_cache.resolve()))

    base_dir = Path(args.base_dir)
    manifest = Path(args.manifest) if args.manifest else None
    items = build_items(base_dir, manifest)
    random.shuffle(items)
    n = len(items)
    test = max(1, round(n * 0.2))
    gen = max(1, round(n * 0.2))
    train_items = items[: n - test - gen]
    test_items = items[n - test - gen : n - gen]
    gen_items = items[n - gen :]
    print(f"总样本 {n}: train={len(train_items)} test={len(test_items)} gen={len(gen_items)}")

    train_ds = MultiLabelDataset(train_items, image_size=args.image_size, augment=True)
    test_ds = MultiLabelDataset(test_items, image_size=args.image_size, augment=False)
    gen_ds = MultiLabelDataset(gen_items, image_size=args.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    gen_loader = DataLoader(gen_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_torchvision_classifier("resnet18", len(MULTILABEL_CLASSES), pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Device: {device}")
    print(f"多标签类别: {MULTILABEL_CLASSES}")

    best_exact = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)

        avg_loss = total_loss / max(seen, 1)
        test_exact, test_pc = evaluate(model, test_loader, device)
        if test_exact > best_exact:
            best_exact = test_exact
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            gen_exact, _ = evaluate(model, gen_loader, device)
            pc_str = " ".join(f"{MULTILABEL_CLASSES[i]}={test_pc[i]:.2f}" for i in range(len(MULTILABEL_CLASSES)))
            print(
                f"epoch {epoch:03d}/{args.epochs} loss={avg_loss:.4f} "
                f"test_exact={test_exact:.3f} [{pc_str}] gen_exact={gen_exact:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    class_to_idx = {name: i for i, name in enumerate(MULTILABEL_CLASSES)}
    save_checkpoint(
        args.out,
        model,
        class_to_idx,
        image_size=args.image_size,
        metadata={"multilabel": True, "best_exact_match": best_exact, "classes": MULTILABEL_CLASSES},
        architecture="resnet18",
    )
    print(f"Saved model: {args.out}")
    print(f"Best exact-match accuracy: {best_exact:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多标签奶龙奶蛙奶蛋分类训练")
    parser.add_argument("--base-dir", default="nailong_naiwa_10_demo", help="单标签数据根目录")
    parser.add_argument("--manifest", default=None, help="多标签标注 json（{路径: [奶龙,奶蛙,奶蛋]}）")
    parser.add_argument("--out", default="models/multilabel_resnet18.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--print-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> int:
    train(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
