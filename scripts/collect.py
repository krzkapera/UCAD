"""Reads the results.csv files a set of runs wrote and prints them three ways.

    python scripts/collect.py runs "ref_visa_*" mean       one line per run
    python scripts/collect.py runs "ref_visa_*" seeds      mean +- sd across the runs
    python scripts/collect.py runs "ref_visa_*" category   per category, with max across the runs

`mean` also prints the second reading every run writes, the `--basic_size` bank in
`results_nolimit/`, which no published table reports but whose distance from the first is what the
README calls FM.

`seeds` groups runs whose names differ only by a trailing seed, so name runs `<config>_s<seed>_<jobid>`
if you want them pooled.

`category` is the one that matters for comparing against the published tables: an average can agree
while every category disagrees, and that is how we first mistook the wrong data split for the right
checkpoint.
"""

import csv
import pathlib
import re
import statistics
import sys

_SEED_SUFFIX = re.compile(r"_s\d+_\d+$|_\d+$")


def _rows(run_dir, results="results"):
    path = run_dir / results / "REFEXP" / "FULL" / "results.csv"
    if not path.exists():
        return None
    return list(csv.DictReader(path.read_text().splitlines()))


def _mean_row(rows):
    means = [r for r in rows if r["Row Names"] == "Mean"]
    return means[-1] if means else None


def report_mean(run_dirs):
    for run_dir in run_dirs:
        rows = _rows(run_dir)
        if rows is None:
            print("%s: no results.csv" % run_dir.name)
            continue
        limited = _mean_row(rows)
        if limited is None:
            print("%s: no Mean row" % run_dir.name)
            continue
        line = "%s: auroc=%.4f pixel_ap=%.4f pixel_auroc=%.4f" % (
            run_dir.name, float(limited["instance_auroc"]),
            float(limited["pixel_ap"]), float(limited["full_pixel_auroc"]),
        )
        unlimited_rows = _rows(run_dir, "results_nolimit")
        unlimited = _mean_row(unlimited_rows) if unlimited_rows else None
        if unlimited:
            line += " | basic_size auroc=%.4f pixel_ap=%.4f" % (
                float(unlimited["instance_auroc"]), float(unlimited["pixel_ap"]),
            )
        print(line)


def report_seeds(run_dirs):
    groups = {}
    for run_dir in run_dirs:
        rows = _rows(run_dir)
        mean = _mean_row(rows) if rows else None
        key = _SEED_SUFFIX.sub("", run_dir.name)
        groups.setdefault(key, []).append(
            None if mean is None else (float(mean["instance_auroc"]), float(mean["pixel_ap"]))
        )

    for key in sorted(groups):
        present = [x for x in groups[key] if x]
        missing = len(groups[key]) - len(present)
        if not present:
            print("%-32s no results (%d runs)" % (key, missing))
            continue
        auroc = [x[0] for x in present]
        pixel = [x[1] for x in present]
        print("%-32s n=%d auroc=%.4f +-%.4f  pixel_ap=%.4f +-%.4f%s" % (
            key, len(present), statistics.mean(auroc),
            statistics.stdev(auroc) if len(auroc) > 1 else 0.0,
            statistics.mean(pixel), statistics.stdev(pixel) if len(pixel) > 1 else 0.0,
            "  missing:%d" % missing if missing else "",
        ))


def report_category(run_dirs):
    per_category = {}
    used = 0
    for run_dir in run_dirs:
        rows = _rows(run_dir)
        if rows is None:
            continue
        used += 1
        for row in (r for r in rows if r["Row Names"] != "Mean"):
            per_category.setdefault(row["Row Names"], []).append(float(row["instance_auroc"]))

    print("runs=%d" % used)
    means, maxes = [], []
    for category in sorted(per_category):
        values = per_category[category]
        means.append(statistics.mean(values))
        maxes.append(max(values))
        print("%-18s n=%2d mean=%.4f sd=%.4f min=%.4f max=%.4f" % (
            category, len(values), statistics.mean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0, min(values), max(values),
        ))
    if means:
        print("%-18s      mean-of-means=%.4f  mean-of-maxes=%.4f  difference=%+.4f" % (
            "AVERAGE", statistics.mean(means), statistics.mean(maxes),
            statistics.mean(maxes) - statistics.mean(means),
        ))


def main():
    if len(sys.argv) != 4 or sys.argv[3] not in ("mean", "seeds", "category"):
        sys.exit(__doc__)
    run_dirs = sorted(pathlib.Path(sys.argv[1]).glob(sys.argv[2]))
    if not run_dirs:
        sys.exit("no run directories match %r" % sys.argv[2])
    {"mean": report_mean, "seeds": report_seeds, "category": report_category}[sys.argv[3]](run_dirs)


if __name__ == "__main__":
    main()
