"""Scoring for BCSA3 spot deconvolution, matching the DestVI/scDOT benchmark."""
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, accuracy_score


def score(prop, clones, spot_clone_numbat, spot_soft, label=""):
    """prop: [n_spots, K] clone proportions. Returns the benchmark's three metrics."""
    prop = np.asarray(prop, dtype=float)
    prop = prop / np.maximum(prop.sum(1, keepdims=True), 1e-12)
    pred = np.asarray(clones)[prop.argmax(1)]
    truth = np.asarray(spot_clone_numbat)

    acc = accuracy_score(truth, pred)
    ari = adjusted_rand_score(truth, pred)
    majority = np.bincount(truth).max() / len(truth)

    # soft agreement: per-clone Pearson against Numbat's posterior
    cors = []
    for k in range(prop.shape[1]):
        t = np.asarray(spot_soft)[:, k]
        if np.std(t) > 1e-8 and np.std(prop[:, k]) > 1e-8:
            cors.append(pearsonr(prop[:, k], t)[0])
    mean_r = float(np.mean(cors)) if cors else np.nan

    out = {"method": label, "argmax_acc": acc, "majority": majority,
           "ari": ari, "mean_clone_pearson": mean_r}
    return out


def fmt(rows):
    hdr = f"{'method':<28} {'acc':>7} {'major':>7} {'ARI':>8} {'clone r':>9}"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(f"{r['method']:<28} {r['argmax_acc']:>7.3f} {r['majority']:>7.3f} "
                     f"{r['ari']:>8.3f} {r['mean_clone_pearson']:>9.3f}")
    return "\n".join(lines)
