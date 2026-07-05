"""Contact sheets from extracted PVSG frames — PIL only, no ImageMagick.

Standalone by design: scp just this file to the cluster and run it in any env
with Pillow (the timeline job's env has it). One JPG per video, every Nth frame
tiled in a fixed-width grid, a few hundred KB each — small enough to scp the
whole batch and pick showcase videos locally.

    python make_sheets.py --frames-root $WORK/frames --out sheets \
        1018_6811493102 1014_5476140602 ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

FRAME_EXTS = (".jpg", ".jpeg", ".png")


def make_sheet(frames_dir: Path, out_path: Path, *, every: int = 15,
               cols: int = 8, thumb_w: int = 160) -> int:
    frames = sorted(p for p in frames_dir.iterdir()
                    if p.suffix.lower() in FRAME_EXTS)[::every]
    if not frames:
        return 0
    first = Image.open(frames[0])
    thumb_h = round(first.height * thumb_w / first.width)
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "black")
    draw = ImageDraw.Draw(sheet)
    for k, fp in enumerate(frames):
        x, y = (k % cols) * thumb_w, (k // cols) * thumb_h
        sheet.paste(Image.open(fp).resize((thumb_w, thumb_h)), (x, y))
        draw.text((x + 3, y + 2), fp.stem, fill="yellow")
    sheet.save(out_path, quality=80)
    return len(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="video ids (= frame subdir names)")
    ap.add_argument("--frames-root", required=True)
    ap.add_argument("--out", default="sheets")
    ap.add_argument("--every", type=int, default=15, help="take every Nth frame")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--thumb-w", type=int, default=160)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in args.videos:
        d = Path(args.frames_root) / v
        if not d.is_dir():
            print(f"{v}: no frames dir at {d} — skipped")
            continue
        n = make_sheet(d, out / f"{v}.jpg", every=args.every,
                       cols=args.cols, thumb_w=args.thumb_w)
        print(f"{v}: {n} thumbnails -> {out / f'{v}.jpg'}")


if __name__ == "__main__":
    main()
