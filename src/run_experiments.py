"""
Early detection of at-risk online learners (OULAD) - baseline vs. LSTM.

One command does everything:
    python src/run_experiments.py

1. downloads OULAD if needed and builds weekly click sequences (src/dataset.py);
2. splits students 70 / 10 / 20 (train / validation / test), grouped by student id;
3. for every horizon w = 1 .. 15 weeks, trains
     - "LR total"  : logistic regression on the TOTAL clicks of weeks 1..w (required baseline)
     - "LR weekly" : logistic regression on the week-by-week vector (same information
                     as the LSTM, but without a notion of order)
     - "LSTM"      : small recurrent network reading the weeks one after the other;
4. evaluates accuracy, F1 and ROC-AUC on the test set, and writes figures and tables.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from dataset import N_WEEKS, build_sequences
from download_data import download

SEED = 42
PROJECT_DIR = Path(__file__).resolve().parents[1]
FIG = PROJECT_DIR / "figures"
OUT = PROJECT_DIR / "outputs"
MODELS = PROJECT_DIR / "models"
torch.set_num_threads(4)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class DropoutLSTM(nn.Module):
    """1-layer LSTM (32 hidden units) + linear layer -> probability of risk.

    Input: a sequence of w weeks, each week described by 2 numbers
    (log clicks, active days). The LSTM keeps a small "memory" vector that it
    updates week after week; the last memory is turned into one score.
    """

    def __init__(self, n_features: int = 2, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_last, _) = self.lstm(x)          # h_last: [1, batch, hidden]
        return self.head(self.dropout(h_last[-1])).squeeze(-1)  # logits


def train_lstm(Xtr, ytr, Xva, yva, max_epochs=80, patience=8, verbose=False):
    """Train with Adam + early stopping on the validation ROC-AUC."""
    torch.manual_seed(SEED)
    model = DropoutLSTM(n_features=Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva)
    gen = torch.Generator().manual_seed(SEED)
    best_auc, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=gen)
        total = 0.0
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_auc = roc_auc_score(yva, predict_lstm(model, Xva_t))
        history.append({"epoch": epoch + 1, "train_loss": total / len(perm), "val_auc": val_auc})
        if verbose:
            print(f"    epoch {epoch + 1:2d} loss={total / len(perm):.4f} val_auc={val_auc:.4f}")
        if val_auc > best_auc + 1e-4:
            best_auc, bad = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_lstm(model, X) -> np.ndarray:
    model.eval()
    X = X if isinstance(X, torch.Tensor) else torch.tensor(X)
    return torch.sigmoid(model(X)).numpy()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def scores(y, prob, mask=None) -> dict:
    if mask is not None:
        y, prob = y[mask], prob[mask]
    pred = (prob >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, prob), "n": int(len(y))}


def features(X: np.ndarray, w: int):
    """log1p clicks + active days/7 for the first w weeks."""
    seq = np.stack([np.log1p(X[:, :w, 0]), X[:, :w, 1] / 7.0], axis=-1).astype(np.float32)
    total = np.log1p(X[:, :w, 0].sum(axis=1, keepdims=True))
    flat = seq.reshape(len(seq), -1)
    return seq, total, flat


def split(data):
    """70/10/20 split; one student never appears in two splits."""
    idx = np.arange(len(data["y"]))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    trval, test = next(gss.split(idx, groups=data["group"]))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=SEED)  # 0.125*0.8 = 0.1
    tr, va = next(gss2.split(trval, groups=data["group"][trval]))
    return trval[tr], trval[va], test


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_eda(data):
    out = pd.Series(data["outcome"]).value_counts()[["Distinction", "Pass", "Fail", "Withdrawn"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(out.index, out.values, color=["#2ca02c", "#98df8a", "#ff9896", "#d62728"])
    for i, v in enumerate(out.values):
        ax.text(i, v, f"{v}\n({v / out.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Enrolments")
    ax.set_ylim(0, out.max() * 1.2)
    ax.set_title("Final result (green = success, red = at risk)")
    fig.tight_layout()
    fig.savefig(FIG / "01_final_results.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    weeks = np.arange(1, N_WEEKS + 1)
    colors = {"Distinction": "#2ca02c", "Pass": "#98df8a", "Fail": "#ff9896", "Withdrawn": "#d62728"}
    for name, c in colors.items():
        m = data["outcome"] == name
        ax.plot(weeks, np.median(data["X"][m, :, 0], axis=0), marker="o", color=c,
                label=f"{name} (median)")
    ax.set_xlabel("Week of the course")
    ax.set_ylabel("Clicks in the week")
    ax.set_title("Weekly VLE clicks by final result")
    ax.set_xticks(weeks)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "02_weekly_clicks_by_outcome.png", dpi=150)
    plt.close(fig)


def plot_week_curves(curve: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    style = {"LR total": ("#7f7f7f", "s"), "LR weekly": ("#1f77b4", "^"), "LSTM": ("#d62728", "o")}
    for metric, ax in zip(["roc_auc", "f1"], axes):
        for model, (c, mk) in style.items():
            sub = curve[(curve.model == model) & (curve.population == "all")]
            ax.plot(sub.week, sub[metric], marker=mk, color=c, label=model)
            sub = curve[(curve.model == model) & (curve.population == "still_enrolled")]
            ax.plot(sub.week, sub[metric], marker=mk, color=c, linestyle=":", alpha=0.7,
                    label=f"{model} (still enrolled)")
        ax.set_xlabel("Number of weeks observed")
        ax.set_ylabel(metric.upper().replace("_", "-"))
        ax.set_xticks(range(1, N_WEEKS + 1))
        ax.grid(alpha=0.3)
    axes[0].set_title("ROC-AUC on the test set vs. weeks observed")
    axes[1].set_title("F1 (threshold 0.5) vs. weeks observed")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "03_performance_by_week.png", dpi=150)
    plt.close(fig)


def plot_final(y, probs: dict, history: list):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves after {N_WEEKS} weeks (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "04_roc_week15.png", dpi=150)
    plt.close(fig)

    cm = confusion_matrix(y, (probs["LSTM"] >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Success", "At risk"])
    ax.set_yticks([0, 1], ["Success", "At risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"LSTM confusion matrix (week {N_WEEKS})")
    fig.tight_layout()
    fig.savefig(FIG / "05_confusion_lstm_week15.png", dpi=150)
    plt.close(fig)

    h = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(h.epoch, h.train_loss, "b-o", label="train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss (BCE)", color="b")
    ax2 = ax1.twinx()
    ax2.plot(h.epoch, h.val_auc, "r-s", label="validation AUC")
    ax2.set_ylabel("Validation ROC-AUC", color="r")
    ax1.set_title(f"LSTM training ({N_WEEKS} weeks)")
    fig.tight_layout()
    fig.savefig(FIG / "06_lstm_training_curve.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    for d in (FIG, OUT, MODELS):
        d.mkdir(exist_ok=True)
    np.random.seed(SEED)

    download()
    data = build_sequences()
    X, y, unreg = data["X"], data["y"], data["unreg"]
    print(f"{len(y)} enrolments, at-risk ratio = {y.mean():.3f}")
    plot_eda(data)

    tr, va, te = split(data)
    print(f"train={len(tr)}  val={len(va)}  test={len(te)}  "
          f"(test at-risk ratio {y[te].mean():.3f})")

    rows, final = [], {}
    for w in range(1, N_WEEKS + 1):
        seq, total, flat = features(X, w)
        # students still registered at the end of week w (unregistration day > 7w)
        still = np.isnan(unreg[te]) | (unreg[te] >= 7 * w)
        probs = {}

        lr_total = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        lr_total.fit(total[tr], y[tr])
        probs["LR total"] = lr_total.predict_proba(total[te])[:, 1]

        lr_week = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        lr_week.fit(flat[tr], y[tr])
        probs["LR weekly"] = lr_week.predict_proba(flat[te])[:, 1]

        # standardise LSTM inputs with train statistics
        mu = seq[tr].reshape(-1, 2).mean(0)
        sd = seq[tr].reshape(-1, 2).std(0) + 1e-6
        norm = ((seq - mu) / sd).astype(np.float32)
        model, history = train_lstm(norm[tr], y[tr], norm[va], y[va], verbose=(w == N_WEEKS))
        probs["LSTM"] = predict_lstm(model, norm[te])

        for name, p in probs.items():
            rows.append({"week": w, "model": name, "population": "all", **scores(y[te], p)})
            rows.append({"week": w, "model": name, "population": "still_enrolled",
                         **scores(y[te], p, still)})
        print(f"week {w:2d}: " + "  ".join(
            f"{n} AUC={roc_auc_score(y[te], p):.3f}" for n, p in probs.items())
            + f"  | LSTM epochs={len(history)}")
        if w == N_WEEKS:
            final = {"probs": probs, "history": history}
            torch.save(model.state_dict(), MODELS / "lstm_week15.pt")

    curve = pd.DataFrame(rows)
    curve.to_csv(OUT / "performance_by_week.csv", index=False)
    plot_week_curves(curve)
    plot_final(y[te], final["probs"], final["history"])
    pd.DataFrame(final["history"]).to_csv(OUT / "lstm_training_history_week15.csv", index=False)

    summary = curve[(curve.population == "all") & (curve.week.isin([5, 10, 15]))]
    print("\nMain results (test set):")
    print(summary.round(3).to_string(index=False))
    meta = {"n_enrolments": int(len(y)), "at_risk_ratio": float(y.mean()),
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "runtime_seconds": round(time.time() - t0, 1)}
    (OUT / "run_summary.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
