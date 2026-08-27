"""Materialises VisA's official 1cls split as the folder tree run_ucad.py reads.

    python scripts/visa_official_split.py \
        /data/visa /data/visa-sam-b /data/visa1cls /data/visa1cls-sam-b /data/visa/split_csv/1cls.csv

The loader walks directories and cannot read `split_csv/1cls.csv`, so the split has to exist as
folders. The per-category copy that circulates - all normals in `train/good` except the first 20 per
class - is a different division, and using it costs 0.06 image AUROC and turns candle into noise; see
`REPRODUCTION.md`.

Images are symlinked out of that per-category copy, which keeps the original file names, so a row of
the csv identifies the same picture in both layouts. Links are absolute: a relative target would
resolve against the link's own directory rather than the working directory, which silently produces a
tree of broken links.

A training image is skipped when no SAM label map exists for it. Masks are generated for the folder
copy's own train split, so a handful of images the official split trains on may have none, and the
loss reads one per training image. The count is printed at the end; it should be small, and a run
with `--epochs_num 0` needs no masks at all.
"""

import csv
import pathlib
import sys


def find(source_root, category, name, anomalous):
    subdirs = ["test/bad"] if anomalous else ["train/good", "test/good"]
    for subdir in subdirs:
        candidate = source_root / category / subdir / name
        if candidate.exists():
            return candidate
    return None


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_symlink():
        destination.symlink_to(source.resolve())


def main():
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    source, sam_source, target, sam_target, split_csv = (pathlib.Path(a) for a in sys.argv[1:])

    counts = {}
    missing_image = missing_mask = 0

    for row in csv.DictReader(split_csv.read_text().splitlines()):
        category, split, label = row["object"], row["split"], row["label"]
        name = pathlib.Path(row["image"]).name
        anomalous = label != "normal"

        found = find(source, category, name, anomalous)
        if found is None:
            missing_image += 1
            continue

        if split == "train":
            sam = sam_source / category / "train" / "good" / name
            if not sam.exists():
                missing_mask += 1
                continue
            link(found, target / category / "train" / "good" / name)
            link(sam, sam_target / category / "train" / "good" / name)
            key = (category, "train")
        elif anomalous:
            link(found, target / category / "test" / "bad" / name)
            stem = pathlib.Path(name).stem
            mask = source / category / "ground_truth" / "bad" / (stem + ".png")
            if mask.exists():
                link(mask, target / category / "ground_truth" / "bad" / (stem + ".png"))
            key = (category, "test/bad")
        else:
            link(found, target / category / "test" / "good" / name)
            key = (category, "test/good")

        counts[key] = counts.get(key, 0) + 1

    for (category, split), count in sorted(counts.items()):
        print("%-14s%-12s%d" % (category, split, count))
    print("images not found: %d, training images skipped for a missing SAM mask: %d"
          % (missing_image, missing_mask))


if __name__ == "__main__":
    main()
