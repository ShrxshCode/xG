# make_plots.py
# Turns the results into a table and three pictures, saved in the results/ folder.
#
#   python make_plots.py
#
# Importing train_model runs it once, which trains the models and gives us the
# variables we need below (X_test, y_test, model, xgb_model, and so on).

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from train_model import (X_test, base_columns, base_model, build_features, model,
                         statsbomb_xg, test, xgb_model, y_test)

os.makedirs("results", exist_ok=True)

# One color per model, kept the same in every picture. StatsBomb is gray because it
# is the reference we compare against, not one of our models.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY,
    "text.color": "#0b0b0b", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
})

# Each model's predicted chance of a goal for every test shot.
predictions = {
    "Logistic (no keeper)": base_model.predict_proba(X_test[base_columns])[:, 1],
    "Logistic + keeper": model.predict_proba(X_test)[:, 1],
    "XGBoost": xgb_model.predict_proba(X_test)[:, 1],
    "StatsBomb xG": statsbomb_xg.values,
}
colors = {"Logistic (no keeper)": BLUE, "Logistic + keeper": ORANGE, "XGBoost": AQUA, "StatsBomb xG": GRAY}

# ---- Table: one row per model ----
rows = []
for name, predicted in predictions.items():
    rows.append({
        "model": name,
        "AUC": round(roc_auc_score(y_test, predicted), 3),
        "log_loss": round(log_loss(y_test, predicted), 4),
        "predicted_goals": round(predicted.sum(), 0),
        "real_goals": int(y_test.sum()),
    })
table = pd.DataFrame(rows)
table.to_csv("results/model_results.csv", index=False)
print()
print(table.to_string(index=False))

# ---- Picture 1: model comparison ----
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

# Left: AUC. The bars start at 0.5 because 0.5 is a coin flip, not zero.
for i, row in table.iterrows():
    axes[0].barh(i, row["AUC"] - 0.5, left=0.5, color=colors[row["model"]], height=0.55)
    axes[0].text(row["AUC"] + 0.004, i, f"{row['AUC']:.3f}", va="center")
axes[0].set_yticks(range(len(table)))
axes[0].set_yticklabels(table["model"])
axes[0].invert_yaxis()
axes[0].set_xlim(0.5, 0.85)
axes[0].set_xlabel("AUC (higher is better, 0.5 = coin flip)")
axes[0].set_title("How well it ranks goals above misses", loc="left", fontsize=11)

# Right: log loss. Lower is better, so the shorter bar wins.
for i, row in table.iterrows():
    axes[1].barh(i, row["log_loss"], color=colors[row["model"]], height=0.55)
    axes[1].text(row["log_loss"] + 0.003, i, f"{row['log_loss']:.4f}", va="center")
axes[1].set_yticks(range(len(table)))
axes[1].set_yticklabels([])
axes[1].invert_yaxis()
axes[1].set_xlim(0, 0.34)
axes[1].set_xlabel("Log loss (lower is better)")
axes[1].set_title("How good the probabilities are", loc="left", fontsize=11)
plt.tight_layout()
plt.savefig("results/1_model_comparison.png", dpi=150)
plt.close()

# ---- Picture 2: calibration ----
# Sort the test shots by predicted xG and cut them into 8 equal groups. In each group,
# compare the average prediction with how often a goal really happened.
# A perfect model sits on the diagonal line.
fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.plot([0, 0.5], [0, 0.5], color=GRID, linewidth=2, label="Perfect")
for name in ["Logistic + keeper", "XGBoost", "StatsBomb xG"]:
    predicted = predictions[name]
    order = np.argsort(predicted)
    groups = np.array_split(order, 8)
    avg_predicted = []
    actual_rate = []
    for group in groups:
        avg_predicted.append(predicted[group].mean())
        actual_rate.append(y_test.values[group].mean())
    ax.plot(avg_predicted, actual_rate, marker="o", markersize=6, linewidth=2,
            color=colors[name], label=name)
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 0.5)
ax.set_xlabel("Predicted chance of a goal (xG)")
ax.set_ylabel("How often a goal really happened")
ax.set_title("When a model says 20%, do 20% score?", loc="left", fontsize=11)
ax.grid(color=GRID, linewidth=0.6)
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig("results/2_calibration.png", dpi=150)
plt.close()

# ---- Picture 3: xG map of the pitch ----
# Ask the XGBoost model for the xG of a shot from every point in the attacking half.
# We fix everything except the location: a right-footed, open-play shot, no pressure,
# with the keeper standing on his line in the middle of the goal.
xs = np.arange(78, 120.5, 1.0)       # from 42 yards out up to the goal line
ys = np.arange(0, 80.5, 1.0)
grid_x, grid_y = np.meshgrid(xs, ys)
points = pd.DataFrame({
    "x": grid_x.ravel(), "y": grid_y.ravel(),
    "gk_x": 119.0, "gk_y": 40.0,
    "body_part": "Right Foot", "shot_type": "Open Play",
    "under_pressure": 0, "first_time": 0,
})
grid_xg = xgb_model.predict_proba(build_features(points))[:, 1].reshape(grid_x.shape)

# The goal is drawn at the top. So sideways on the picture is y, and up is x.
fig, ax = plt.subplots(figsize=(8, 5.6))
shown = ax.pcolormesh(grid_y, grid_x, grid_xg, cmap="Blues", shading="auto", vmin=0, vmax=0.5)
colorbar = plt.colorbar(shown, ax=ax, fraction=0.03, pad=0.02)
colorbar.set_label("xG (chance of a goal)")

# Pitch lines, drawn in dark gray on top of the colors.
line = {"color": GRAY, "linewidth": 1.4}
ax.plot([0, 80], [120, 120], **line)                                   # goal line
ax.plot([18, 18, 62, 62], [120, 102, 102, 120], **line)                # penalty box
ax.plot([30, 30, 50, 50], [120, 114, 114, 120], **line)                # six-yard box
ax.plot([36, 36, 44, 44], [120, 122, 122, 120], **line)                # the goal
ax.scatter([40], [108], color=GRAY, s=14)                              # penalty spot
ax.scatter([40], [119], color="#0b0b0b", s=40, zorder=3)               # the keeper
ax.text(42, 118.2, "keeper", fontsize=9, color=GRAY, va="center")
ax.set_xlim(0, 80)
ax.set_ylim(78, 123)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("XGBoost xG by shot location\nright foot, open play, no pressure, keeper on his line",
             loc="left", fontsize=11)
plt.tight_layout()
plt.savefig("results/3_xg_pitch_map.png", dpi=150)
plt.close()

print()
print("saved results/model_results.csv and three pictures in results/")
