"""Draws docs/architecture.png (run: python docs/make_architecture.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(11, 4.2)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 4.2)
def box(x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=color, ec="#1F3864", lw=1.4))
    ax.text(x + w/2, y + h*0.64, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#14213d")
    ax.text(x + w/2, y + h*0.30, sub, ha="center", va="center", fontsize=8.2, color="#333")
def arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#1F3864", lw=1.6))
    if label:
        if x1 == x2: ax.text(x1 + 0.12, (y1+y2)/2, label, ha="left", va="center", fontsize=8, color="#1F3864")
        else: ax.text((x1+x2)/2, (y1+y2)/2 + 0.16, label, ha="center", fontsize=8, color="#1F3864")
ax.text(2.7, 3.95, "TRAINING (offline)", ha="center", fontsize=10, color="#1F3864", fontweight="bold")
ax.text(8.2, 3.95, "SERVING (online)", ha="center", fontsize=10, color="#1F3864", fontweight="bold")
box(0.1, 2.2, 1.9, 1.3, "titanic.csv", "raw data\n891 passengers", "#E8EEF9")
box(2.5, 2.2, 2.5, 1.3, "ml/train.py", "split, CV, tuning,\nselect best model", "#D9E2F3")
box(2.5, 0.35, 2.5, 1.3, "sklearn Pipeline", "features + impute + scale\n+ encode + model", "#D9E2F3")
box(5.5, 0.35, 2.0, 1.3, ".joblib file", "saved pipeline\n+ metadata.json", "#FFF2CC")
box(5.5, 2.2, 2.0, 1.3, "app/main.py", "FastAPI, loads model\nonce at startup", "#E2F0D9")
box(8.0, 2.2, 2.9, 1.3, "Pydantic validation", "types, ranges,\nallowed values", "#E2F0D9")
box(8.0, 0.35, 2.9, 1.3, "Client", "web page /  curl /\nSwagger UI (/docs)", "#F8E1D9")
arrow(2.0, 2.85, 2.5, 2.85); arrow(3.75, 2.2, 3.75, 1.65); arrow(5.0, 1.0, 5.5, 1.0, "joblib.dump")
arrow(6.5, 1.65, 6.5, 2.2, "joblib.load"); arrow(8.0, 2.85, 7.5, 2.85); arrow(9.45, 1.65, 9.45, 2.2, "JSON request")
plt.tight_layout(); plt.savefig("docs/architecture.png", dpi=150); print("saved")
