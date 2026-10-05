# download_data.py
# Step 1: get shot data from StatsBomb's free open data on GitHub.
# StatsBomb records every event in a match (passes, shots, tackles...). We only
# keep the SHOTS, because a shot is one row: where it was taken, how, and whether
# it was a goal. That is exactly what an xG model learns from.
#
#   python download_data.py          (takes a few minutes, saves data/shots.csv)

import csv
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"

# (competition_id, season_id) pairs. Men's competitions only, so the model
# isn't mixing two different games' shot patterns.
SEASONS = [
    (2, 27),     # Premier League 2015/16
    (12, 27),    # Serie A 2015/16
    (7, 27),     # Ligue 1 2015/16
    (9, 27),     # Bundesliga 2015/16
    (9, 281),    # Bundesliga 2023/24
    (11, 27),    # La Liga 2015/16
    (43, 3),     # World Cup 2018
    (43, 106),   # World Cup 2022
    (55, 43),    # Euro 2020
    (55, 282),   # Euro 2024
]


def get_json(path):
    """Download one JSON file from the open-data repo and parse it.

    The server sometimes drops a connection, so try up to 5 times before giving up."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(BASE + path, timeout=60) as response:
                return json.loads(response.read())
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2)


def shots_from_match(match_id):
    """Download one match's events and return one dict per shot."""
    events = get_json("events/" + str(match_id) + ".json")
    rows = []
    for event in events:
        if event["type"]["name"] != "Shot":
            continue
        shot = event["shot"]
        outcome = shot["outcome"]["name"]
        # Skip penalties (they are not an open-play chance) and shootouts.
        if shot["type"]["name"] == "Penalty" or event["period"] == 5:
            continue
        # The freeze frame lists every visible player at the moment of the shot.
        # We only want the opposing goalkeeper: position "Goalkeeper" and teammate False.
        # If he isn't in the list (or there is no freeze frame) we leave his spot empty.
        gk_x = None
        gk_y = None
        for player in shot.get("freeze_frame", []):
            if player["position"]["name"] == "Goalkeeper" and not player["teammate"]:
                gk_x = player["location"][0]
                gk_y = player["location"][1]

        # location is [x, y] on a pitch 120 long and 80 wide.
        # The goal we attack is always at x = 120.
        rows.append({
            "match_id": match_id,
            "x": event["location"][0],
            "y": event["location"][1],
            "body_part": shot["body_part"]["name"],
            "shot_type": shot["type"]["name"],
            "technique": shot["technique"]["name"],
            "under_pressure": 1 if event.get("under_pressure") else 0,
            "first_time": 1 if shot.get("first_time") else 0,
            "minute": event["minute"],
            "gk_x": gk_x,
            "gk_y": gk_y,
            "sb_xg": shot.get("statsbomb_xg"),   # StatsBomb's own xG, for comparison
            "goal": 1 if outcome == "Goal" else 0,
        })
    return rows


def main():
    # 1. Collect every match id for the seasons we want.
    match_ids = []
    for competition_id, season_id in SEASONS:
        matches = get_json("matches/" + str(competition_id) + "/" + str(season_id) + ".json")
        for match in matches:
            match_ids.append(match["match_id"])
    print("matches to download:", len(match_ids))

    # 2. Download them several at a time, since each is a separate web request.
    all_rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for rows in pool.map(shots_from_match, match_ids):
            all_rows = all_rows + rows

    # 3. Save to a CSV file.
    with open("data/shots.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    goals = sum(row["goal"] for row in all_rows)
    print("shots:", len(all_rows), "goals:", goals, "rate:", round(goals / len(all_rows), 3))
    with_keeper = sum(1 for row in all_rows if row["gk_x"] is not None)
    print("shots with a goalkeeper position:", with_keeper)


main()
