"""Contact sheets straight from PVSG mp4s — PIL + imageio only, no ImageMagick.

There are no extracted frame JPGs on the cluster (`precompute_all` decodes mp4s
on the fly), so this samples the video the same way the feature extractor does:
a 5-FPS subsample paired 1:1 with the sorted mask list. The number stamped on
each thumbnail is therefore the *annotated frame index* — the same axis the
spans, boundaries, and timelines use, so a good moment on a sheet can be quoted
directly as (video, frame) for the showcase dump.

Standalone by design: scp just this file and run it in the extraction venv
(has Pillow + imageio-ffmpeg).

    python make_sheets.py --videos $WORK/pvsg/VidOR/videos \
        --masks $WORK/pvsg/VidOR/masks --out sheets 1018_6811493102 ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def find_video(videos_dir: Path, vid: str) -> Path | None:
    hits = sorted(videos_dir.glob(f"{vid}.*"))
    return hits[0] if hits else None


def annotated_frames(mp4: Path, n_masks: int):
    """Yield (annotated_frame_index, frame_uint8_HWC), mirroring
    `precompute_all.frames_for_masks` (frame 0 → mask 0, every round(fps/5)th)."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(mp4), "ffmpeg")
    fps = reader.get_meta_data().get("fps") or 30
    step = max(1, round(fps / 5.0))
    mi = 0
    for i, frame in enumerate(reader):
        if i % step == 0:
            yield mi, frame
            mi += 1
            if mi >= n_masks:
                break
    reader.close()


def make_sheet(mp4: Path, n_masks: int, out_path: Path, *, every: int = 15,
               cols: int = 8, thumb_w: int = 160) -> int:
    thumbs: list[tuple[int, Image.Image]] = []
    for mi, frame in annotated_frames(mp4, n_masks):
        if mi % every:
            continue
        img = Image.fromarray(frame)
        thumb_h = round(img.height * thumb_w / img.width)
        thumbs.append((mi, img.resize((thumb_w, thumb_h))))
    if not thumbs:
        return 0
    thumb_h = thumbs[0][1].height
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "black")
    draw = ImageDraw.Draw(sheet)
    for k, (mi, img) in enumerate(thumbs):
        x, y = (k % cols) * thumb_w, (k // cols) * thumb_h
        sheet.paste(img, (x, y))
        draw.text((x + 3, y + 2), str(mi), fill="yellow")
    sheet.save(out_path, quality=80)
    return len(thumbs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="+")
    ap.add_argument("--videos", required=True, help="dir with <video_id>.mp4")
    ap.add_argument("--masks", required=True, help="dir with <video_id>/*.png")
    ap.add_argument("--out", default="sheets")
    ap.add_argument("--every", type=int, default=15,
                    help="take every Nth annotated frame")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--thumb-w", type=int, default=160)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for vid in args.video_ids:
        mp4 = find_video(Path(args.videos), vid)
        n_masks = len(list((Path(args.masks) / vid).glob("*.png")))
        if mp4 is None or n_masks == 0:
            print(f"{vid}: missing mp4 or masks — skipped")
            continue
        n = make_sheet(mp4, n_masks, out / f"{vid}.jpg", every=args.every,
                       cols=args.cols, thumb_w=args.thumb_w)
        print(f"{vid}: {n} thumbnails -> {out / f'{vid}.jpg'}")


if __name__ == "__main__":
    main()
