# train_model.py
# Builds an xG (expected goals) model: given a shot, predict the chance it is a goal.
#
# Steps, in order:
#   1. load the shots
#   2. split into a training set and a test set
#   3. build features (distance, angle, body part, ...)
#   4. train a logistic regression
#   5. evaluate on the test set and compare with StatsBomb's own xG
#
#   python train_model.py

import math

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

# The pitch is 120 x 80. The goal is at x = 120, centered at y = 40, with posts
# at y = 36 and y = 44 (the goal is 8 wide).
GOAL_X = 120
POST_LEFT = 36
POST_RIGHT = 44
GOAL_WIDTH = 8


def distance_to_goal(x, y):
    """Straight-line distance from the shot to the middle of the goal (Pythagoras)."""
    return math.sqrt((GOAL_X - x) ** 2 + (40 - y) ** 2)


def shot_angle(x, y):
    """Angle in degrees at the shooter between the two goal posts.

    A bigger angle means more of the goal is visible. We know all 3 sides of a
    triangle: the distance to each post (Pythagoras) and the goal width. The law
    of cosines turns three sides into an angle:
        cos(angle) = (a*a + b*b - c*c) / (2*a*b)"""
    a = math.sqrt((GOAL_X - x) ** 2 + (POST_LEFT - y) ** 2)    # distance to one post
    b = math.sqrt((GOAL_X - x) ** 2 + (POST_RIGHT - y) ** 2)   # distance to the other
    c = GOAL_WIDTH
    a = max(a, 0.01)    # standing exactly on a post would make a = 0 and divide by zero
    b = max(b, 0.01)
    cos_angle = (a * a + b * b - c * c) / (2 * a * b)
    cos_angle = max(-1, min(1, cos_angle))   # guard against tiny rounding errors
    return math.degrees(math.acos(cos_angle))


def keeper_offset(x, y, gk_x, gk_y):
    """How far sideways the keeper is from the straight line between the shooter
    and the middle of the goal. 0 means he is standing right on the line.

    This is the perpendicular distance from a point to a line:
        |cross product| / length of the line"""
    line_x = GOAL_X - x          # the line from the shooter to the middle of the goal
    line_y = 40 - y
    to_keeper_x = gk_x - x       # the line from the shooter to the keeper
    to_keeper_y = gk_y - y
    cross = line_x * to_keeper_y - line_y * to_keeper_x
    length = math.sqrt(line_x ** 2 + line_y ** 2)
    length = max(length, 0.01)    # a shot from the middle of the goal would make this 0
    return abs(cross) / length


def keeper_distance(x, y, gk_x, gk_y):
    """Straight-line distance from the shooter to the keeper (Pythagoras)."""
    return math.sqrt((gk_x - x) ** 2 + (gk_y - y) ** 2)


def build_features(shots):
    """Turn the raw shot table into the numbers the model learns from."""
    features = pd.DataFrame()

    # Location features. We loop over the rows so each step is easy to follow.
    distances = []
    angles = []
    offsets = []
    keeper_distances = []
    for i in range(len(shots)):
        x = shots["x"].iloc[i]
        y = shots["y"].iloc[i]
        distances.append(distance_to_goal(x, y))
        angles.append(shot_angle(x, y))
        gk_x = shots["gk_x"].iloc[i]
        gk_y = shots["gk_y"].iloc[i]
        offsets.append(keeper_offset(x, y, gk_x, gk_y))
        keeper_distances.append(keeper_distance(x, y, gk_x, gk_y))
    features["distance"] = distances
    features["angle"] = angles
    features["gk_offset"] = offsets
    features["gk_distance"] = keeper_distances

    # Yes/no features become 1 or 0.
    features["is_header"] = (shots["body_part"] == "Head").astype(int).values
    features["is_free_kick"] = (shots["shot_type"] == "Free Kick").astype(int).values
    features["under_pressure"] = shots["under_pressure"].values
    features["first_time"] = shots["first_time"].values
    return features


# 1. Load the shots.
shots = pd.read_csv("data/shots.csv")
shots = shots.dropna(subset=["gk_x", "gk_y"])   # drop the few shots with no keeper position
print("shots:", len(shots), "| goal rate:", round(shots["goal"].mean(), 3))

# 2. Split by MATCH. The model trains on 80% of the matches and we grade it on the
#    other 20%. Splitting by match (not by shot) stops shots from one game, which
#    share teams, players and goalkeepers, from landing in both sets.
#    random_state makes the split repeatable.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_rows, test_rows = next(splitter.split(shots, groups=shots["match_id"]))
train = shots.iloc[train_rows]
test = shots.iloc[test_rows]
print("train shots:", len(train), "| test shots:", len(test))

# 3. Features.
X_train = build_features(train)
X_test = build_features(test)
y_train = train["goal"]
y_test = test["goal"]

# 4. Train two models on the same shots so we can see what the keeper adds.
#    "base" uses the original features. "with keeper" also uses the two keeper ones.
#    max_iter gives the model enough steps to settle on its weights.
keeper_columns = ["gk_offset", "gk_distance"]
base_columns = [c for c in X_train.columns if c not in keeper_columns]

base_model = LogisticRegression(max_iter=1000)
base_model.fit(X_train[base_columns], y_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# XGBoost builds many small decision trees, where each new tree tries to fix the
# mistakes the earlier ones made. Unlike logistic regression, it can learn effects
# that depend on each other, like "a keeper off his line matters more up close".
# These settings keep the trees small and the learning slow so it does not memorize
# the training shots. We picked them up front and did NOT tune them on the test set.
xgb_model = XGBClassifier(
    n_estimators=300,      # number of trees
    max_depth=3,           # how many questions each tree can ask
    learning_rate=0.05,    # how much each tree is allowed to correct
    subsample=0.8,         # each tree sees a random 80% of the shots
    random_state=42,
)
xgb_model.fit(X_train, y_train)

# What did the keeper model learn? A positive number means "raises the chance of a goal".
print()
print("what the model learned (positive = more likely a goal):")
for name, weight in zip(X_train.columns, model.coef_[0]):
    print(f"  {name:<15}{weight:>8.3f}")

# 5. Evaluate. predict_proba gives [chance of no goal, chance of goal] per shot.
base_xg = base_model.predict_proba(X_test[base_columns])[:, 1]
our_xg = model.predict_proba(X_test)[:, 1]
xgb_xg = xgb_model.predict_proba(X_test)[:, 1]
statsbomb_xg = test["sb_xg"]
baseline = [y_train.mean()] * len(y_test)   # guessing the average goal rate every time

# Log loss: lower is better (punishes confident wrong answers).
# AUC: 0.5 is a coin flip, 1.0 is perfect at ranking goals above misses.
print()
print("model                 log loss     AUC")
print(f"always the average    {log_loss(y_test, baseline):>8.4f}   0.500")
print(f"base (no keeper)      {log_loss(y_test, base_xg):>8.4f}   {roc_auc_score(y_test, base_xg):.3f}")
print(f"with keeper           {log_loss(y_test, our_xg):>8.4f}   {roc_auc_score(y_test, our_xg):.3f}")
print(f"XGBoost               {log_loss(y_test, xgb_xg):>8.4f}   {roc_auc_score(y_test, xgb_xg):.3f}")
print(f"StatsBomb xG          {log_loss(y_test, statsbomb_xg):>8.4f}   {roc_auc_score(y_test, statsbomb_xg):.3f}")

# Which features did XGBoost lean on most? (higher = used more)
print()
print("XGBoost feature importance:")
for name, score in sorted(zip(X_train.columns, xgb_model.feature_importances_), key=lambda pair: -pair[1]):
    print(f"  {name:<15}{score:>8.3f}")

# Sanity check: total predicted goals should be close to total real goals.
print()
print("real goals in test set:", y_test.sum(), "| logistic xG total:", round(our_xg.sum(), 1),
      "| XGBoost total:", round(xgb_xg.sum(), 1),
      "| StatsBomb total:", round(statsbomb_xg.sum(), 1))

# Trace the keeper examples from earlier. Both shots are from (108, 40).
print()
print("keeper at (119, 40): offset", round(keeper_offset(108, 40, 119, 40), 1),
      "| distance", round(keeper_distance(108, 40, 119, 40), 1))
print("keeper at (108, 51): offset", round(keeper_offset(108, 40, 108, 51), 1),
      "| distance", round(keeper_distance(108, 40, 108, 51), 1))

# Trace the two example shots from earlier.
print()
print("angle at (108, 40):", round(shot_angle(108, 40), 1), "degrees")
print("angle at (108, 10):", round(shot_angle(108, 10), 1), "degrees")
