"""Applies the reported protocol over seeds instead of epochs, from dumped score vectors.

    UCAD_DUMP_SCORES=scores/visa1cls_s0 ... one run per seed ...
    python scripts/seed_ensemble.py scores visa1cls 25

Each run writes one `<concept>_ep<NN>.npz` per concept and epoch holding that evaluation's raw
per-image score vector and its labels. With `--epochs_num 0 UCAD_EVAL_UNTRAINED=1` there is exactly
one file per concept, so a set of seeds gives one independent draw of the coreset each - the same
thing the epoch loop produces, without any training in the way.

Four columns, per concept:

    mean of AUROCs      each seed scored on its own, then averaged - the honest reading
    best of AUROCs      the selection half of the protocol, applied across seeds
    AUROC of the mean   the ensemble half: rescale each vector, average element-wise, score once
    whole protocol      both, exactly as the code layers them - the running mean's best prefix

An AUROC cannot be averaged back into the ensemble, which is why this needs the vectors and not
`results.csv`.
"""

import glob
import os
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


def _rescale(vector):
    low, high = vector.min(), vector.max()
    return (vector - low) / (high - low) if high > low else np.zeros_like(vector)


def _load(root, dataset, seeds):
    concepts = {}
    for seed in range(seeds):
        pattern = os.path.join(root, "%s_s%d" % (dataset, seed), "*_ep-1.npz")
        for path in sorted(glob.glob(pattern)):
            concept = re.sub(r"_ep-?\d+\.npz$", "", os.path.basename(path))
            data = np.load(path)
            concepts.setdefault(concept, {})[seed] = (
                data["scores"].astype(np.float64), data["labels"].astype(int),
            )
    return concepts


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    root, dataset, seeds = sys.argv[1], sys.argv[2], int(sys.argv[3])

    concepts = _load(root, dataset, seeds)
    if not concepts:
        sys.exit("no score dumps under %s for %s" % (root, dataset))

    print("%-18s %10s %10s %10s %10s  %s" % (
        "concept", "mean", "best", "ensemble", "protocol", "seeds"))
    columns = {name: [] for name in ("mean", "best", "ensemble", "protocol")}

    for concept in sorted(concepts):
        draws = [concepts[concept][seed] for seed in sorted(concepts[concept])]
        labels = draws[0][1]
        aurocs = [roc_auc_score(labels, scores) for scores, _ in draws]

        running = np.zeros_like(draws[0][0])
        prefixes = []
        for index, (scores, _) in enumerate(draws):
            running = running + _rescale(scores)
            prefixes.append(roc_auc_score(labels, running / (index + 1)))

        row = (float(np.mean(aurocs)), float(np.max(aurocs)), prefixes[-1], float(np.max(prefixes)))
        for name, value in zip(("mean", "best", "ensemble", "protocol"), row):
            columns[name].append(value)
        print("%-18s %10.4f %10.4f %10.4f %10.4f  %d" % (concept, *row, len(draws)))

    print("%-18s %10.4f %10.4f %10.4f %10.4f" % (
        "AVERAGE", *[float(np.mean(columns[name]))
                     for name in ("mean", "best", "ensemble", "protocol")]))


if __name__ == "__main__":
    main()
