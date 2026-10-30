import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

root = Path(__file__).resolve().parent
scorecard = json.loads((root / "scorecard.json").read_text())
metrics = scorecard["metrics"]
values = np.array([item["score"] for item in metrics]).reshape(5, 6)
labels = [
    "Completeness", "Correctness", "Depth", "Workflow coverage", "Consistency", "Interoperability",
    "Architecture", "Code quality", "Extensibility", "Dependencies", "State integrity", "Maintainability",
    "Conformance", "API design", "Protocol interop", "Discoverability", "Compatibility", "Source fidelity",
    "Reliability", "Recovery", "Security", "Data safety", "Performance", "Scalability",
    "Developer experience", "Documentation", "Validation", "Observability", "Differentiation", "Innovation potential*",
]
fig, ax = plt.subplots(figsize=(15, 7.5))
fig.subplots_adjust(left=0.15, right=0.92, top=0.85, bottom=0.17)
image = ax.imshow(values, cmap="YlGnBu", vmin=1, vmax=10, aspect="auto")
for index, metric in enumerate(metrics):
    row, column = divmod(index, 6)
    color = "white" if metric["score"] >= 7 else "#10212c"
    ax.text(column, row - 0.25, metric["id"], ha="center", va="center", color=color, fontsize=10)
    ax.text(column, row, str(metric["score"]), ha="center", va="center", color=color, fontsize=22, fontweight="bold")
    ax.text(column, row + 0.27, labels[index], ha="center", va="center", color=color, fontsize=8.5)
ax.set_xticks([])
ax.set_yticks(range(5), ["Functional · 25%", "Engineering · 20%", "Standards · 20%", "Reliability · 25%", "Maturity · 10%"], fontsize=10)
ax.tick_params(axis="y", length=0, pad=10)
ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.add_patch(Rectangle((4.52, 3.52), 0.96, 0.96, fill=False, edgecolor="#f8c75c", linewidth=2.5, linestyle="--"))
cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, ticks=range(1, 11))
cbar.ax.set_ylabel("Quality score", rotation=90)
fig.suptitle("Holographic Memory quality scorecard", fontsize=19, fontweight="bold", y=0.97)
fig.text(0.5, 0.905, "Current quality 7.0/10 · Confidence 3.9/5 · Source 2e4c91d4 · 30 October 2026", ha="center", fontsize=11)
fig.text(0.15, 0.10, "* M6 is forward-looking and excluded from the weighted score. No current rating is 9 or 10.", fontsize=10)
fig.text(0.15, 0.06, "Ratings are engineering judgments. Five reproduced defects and scoped release blockers remain; see REPORT.md.", fontsize=9)
fig.savefig(root / "scorecard.svg")
fig.savefig(root / "scorecard.png", dpi=140)
plt.close(fig)
