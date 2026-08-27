"""Why the loss can win under the reported protocol while losing as a single model.

    UCAD_DUMP_SCORES=scores_ep/code_visa1cls_s3 ... a 25-epoch run ...
    python scripts/epoch_diversity.py scores_ep/code_visa1cls_s3

The protocol averages the 25 epochs' rescaled score vectors. An average helps more when its members
disagree, so a loss that keeps moving the prompt can improve the ensemble while making every member
worse. This measures both halves of that: how correlated the members are, and what the plain
ensemble - the average, with no epoch selection on top - is worth against the last epoch alone.

Per concept, then averaged:

    corr            mean Pearson correlation between pairs of epoch score vectors
    last            AUROC of the last epoch on its own
    ensemble        AUROC of the mean of all 25 rescaled vectors
    gain            ensemble - last, what averaging buys

Pass several directories to print one row each.
"""

import glob
import os
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

_EPOCH = re.compile(r"_ep(-?\d+)\.npz$")


def _rescale(vector):
    low, high = vector.min(), vector.max()
    return (vector - low) / (high - low) if high > low else np.zeros_like(vector)


def _concepts(run_dir):
    grouped = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.npz"))):
        match = _EPOCH.search(os.path.basename(path))
        if not match:
            continue
        concept = _EPOCH.sub("", os.path.basename(path))
        grouped.setdefault(concept, {})[int(match.group(1))] = path
    return grouped


def analyse(run_dir):
    corrs, lasts, ensembles = [], [], []
    for concept, by_epoch in sorted(_concepts(run_dir).items()):
        epochs = sorted(e for e in by_epoch if e >= 0)
        if len(epochs) < 2:
            continue
        vectors, labels = [], None
        for epoch in epochs:
            data = np.load(by_epoch[epoch])
            vectors.append(_rescale(data["scores"].astype(np.float64)))
            labels = data["labels"].astype(int)

        stacked = np.vstack(vectors)
        correlation = np.corrcoef(stacked)
        off_diagonal = ~np.eye(len(vectors), dtype=bool)
        corrs.append(float(np.nanmean(correlation[off_diagonal])))
        lasts.append(roc_auc_score(labels, vectors[-1]))
        ensembles.append(roc_auc_score(labels, stacked.mean(axis=0)))

    if not corrs:
        return None
    return float(np.mean(corrs)), float(np.mean(lasts)), float(np.mean(ensembles))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("%-34s %8s %8s %10s %8s" % ("run", "corr", "last", "ensemble", "gain"))
    for run_dir in sys.argv[1:]:
        result = analyse(run_dir)
        if result is None:
            print("%-34s no usable dumps" % os.path.basename(run_dir.rstrip("/")))
            continue
        corr, last, ensemble = result
        print("%-34s %8.4f %8.4f %10.4f %+8.4f" % (
            os.path.basename(run_dir.rstrip("/")), corr, last, ensemble, ensemble - last))


if __name__ == "__main__":
    main()
