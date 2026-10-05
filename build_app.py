# build_app.py
# Builds the clickable pitch web page (option 1: precomputed answers).
#
# A web page cannot run XGBoost, so we run it here ahead of time. For every spot
# on the pitch (one point per yard) and every combination of the page's settings,
# we ask the model for its xG and save the answers inside the page. When someone
# clicks, the page just looks the answer up.
#
#   python build_app.py        (writes app/index.html, takes about a minute)

import json
import os

import numpy as np
import pandas as pd

# Importing train_model trains the models and gives us xgb_model, build_features
# and shots. (It also prints its usual results.)
from train_model import build_features, shots, xgb_model

# The grid of spots we precompute: x from 64 up to the goal line, y across the width.
XS = list(range(64, 121))
YS = list(range(0, 81))

# The keeper positions the page offers. [x, y] on the same 120 x 80 pitch.
# On the page, y = 0 is the left side and y = 80 the right side.
KEEPERS = [
    ("Set on his line", 119, 40),
    ("Coming out, 4 yards", 115, 40),
    ("Rushing out, 10 yards", 109, 40),
    ("Shaded to the left post", 119, 36),
    ("Shaded to the right post", 119, 44),
]

# Every spot on the grid as a table of shots (one row per spot).
spots = pd.DataFrame({
    "x": [x for x in XS for y in YS],
    "y": [y for x in XS for y in YS],
})

# Ask the model for the xG of every spot under every combination of settings.
grids = {}
for kind in ["foot", "head"]:
    for pressure in ["free", "press"]:
        for k in range(len(KEEPERS)):
            table = spots.copy()
            table["gk_x"] = KEEPERS[k][1]
            table["gk_y"] = KEEPERS[k][2]
            table["body_part"] = "Head" if kind == "head" else "Right Foot"
            table["shot_type"] = "Open Play"
            table["under_pressure"] = 1 if pressure == "press" else 0
            table["first_time"] = 0
            xg = xgb_model.predict_proba(build_features(table))[:, 1]
            # Save as whole numbers out of 1000 (0.234 becomes 234) to keep the page small.
            grids[kind + "|" + pressure + "|" + str(k)] = [int(round(v * 1000)) for v in xg]
            print("done", kind, pressure, KEEPERS[k][0])

# For every spot, count the real shots taken within 3 yards. Where this number is
# small the model is mostly guessing, so the page fades those spots.
shot_x = shots["x"].values
shot_y = shots["y"].values
nearby = []
for x in XS:
    for y in YS:
        close = (shot_x - x) ** 2 + (shot_y - y) ** 2 <= 9    # 3 yards squared
        nearby.append(int(close.sum()))

# Numbers for the page's footer, read from the results table made by make_plots.py.
meta = {
    "shots": int(len(shots)),
    "matches": int(shots["match_id"].nunique()),
    "goal_rate": round(float(shots["goal"].mean()), 4),
    "auc": None,
    "sb_auc": None,
}
if os.path.exists("results/model_results.csv"):
    results = pd.read_csv("results/model_results.csv").set_index("model")
    meta["auc"] = float(results.loc["XGBoost", "AUC"])
    meta["sb_auc"] = float(results.loc["StatsBomb xG", "AUC"])

data = {
    "meta": meta,
    "x0": XS[0], "x1": XS[-1], "ny": len(YS),
    "keepers": [{"name": n, "x": x, "y": y} for n, x, y in KEEPERS],
    "grids": grids,
    "nearby": nearby,
}

# Put the data into the page template and save the finished page.
template = open("app/template.html", encoding="utf-8").read()
page = template.replace("__DATA__", json.dumps(data, separators=(",", ":")))
with open("app/index.html", "w", encoding="utf-8") as f:
    f.write(page)
print("wrote app/index.html,", round(len(page) / 1024), "KB")
