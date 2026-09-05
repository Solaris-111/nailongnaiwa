# -*- coding: utf-8 -*-
"""生成四类（nailong/naiwa/naidan/unrelated）的 train/test/generalization 划分。"""
import random, shutil
from pathlib import Path

SEED = 23
BASE = Path('nailong_naiwa_10_demo')
OUT = Path('nailong_naiwa_splits_4class')
PER_CLASS = 150
SOURCES = {
    'nailong': BASE / 'nailong',
    'naiwa': BASE / 'naiwa',
    'naidan': BASE / 'naidan',
    'unrelated': BASE / 'unrelated',
}
SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}

rng = random.Random(SEED)
if OUT.exists():
    shutil.rmtree(OUT)

for cls, src in SOURCES.items():
    images = sorted(p for p in src.iterdir() if p.suffix.lower() in SUFFIXES)
    if len(images) > PER_CLASS:
        images = rng.sample(images, PER_CLASS)
    images = sorted(images, key=lambda p: p.name)
    rng.shuffle(images)
    n = len(images)
    test = max(1, round(n * 0.2))
    gen = max(1, round(n * 0.2))
    train = n - test - gen
    splits = {
        'train': images[:train],
        'test': images[train:train + test],
        'generalization': images[train + test:],
    }
    for split, paths in splits.items():
        d = OUT / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, d / p.name)
    print(f'{cls}: total={n}  train={train}  test={test}  gen={gen}')

print('done ->', OUT)
