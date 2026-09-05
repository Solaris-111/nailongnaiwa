from ultralytics import YOLO
from pathlib import Path
import shutil

BEST = 'runs/detect/nailong_detection/runs/yolov8n_v1-3/weights/best.pt'
EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
PREFIXES = ['nailong_', 'naiwa_', 'naidan_', 'cooccur_']
SRC = {
    'nailong': 'nailong_naiwa_10_demo/nailong',
    'naiwa': 'nailong_naiwa_10_demo/naiwa',
    'naidan': 'nailong_naiwa_10_demo/naidan',
}
COOCCUR = 'nailong_naiwa_10_demo/multilabel_naiwa_naidan'


def collect_selected() -> set[str]:
    selected = set()
    for d in ['nailong_detection/images/train', 'nailong_detection/images/val']:
        for p in Path(d).iterdir():
            for prefix in PREFIXES:
                if p.stem.startswith(prefix):
                    selected.add(p.stem[len(prefix):])
                    break
    return selected


if __name__ == '__main__':
    model = YOLO(BEST)
    selected = collect_selected()

    out_img = Path('nailong_detection/images/prelabel')
    out_lbl = Path('nailong_detection/labels/prelabel')
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    remaining = []
    for src in list(SRC.values()) + [COOCCUR]:
        for p in Path(src).iterdir():
            if p.suffix.lower() in EXTS and p.stem not in selected:
                remaining.append(p)

    print(f'剩余图 {len(remaining)} 张，开始预标注...')
    n_box = 0
    n_empty = 0
    for p in remaining:
        dst = out_img / p.name
        shutil.copy2(p, dst)
        results = model.predict(source=str(dst), conf=0.2, verbose=False)
        boxes = results[0].boxes
        txt = out_lbl / (dst.stem + '.txt')
        if boxes is not None and len(boxes) > 0:
            cls = boxes.cls.cpu().numpy()
            xywhn = boxes.xywhn.cpu().numpy()
            lines = [f'{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}'
                     for c, (x, y, w, h) in zip(cls, xywhn)]
            txt.write_text('\n'.join(lines), encoding='utf-8')
            n_box += len(lines)
        else:
            txt.write_text('', encoding='utf-8')
            n_empty += 1

    print(f'完成：{len(remaining)} 张，共 {n_box} 个框，{n_empty} 张无框')
