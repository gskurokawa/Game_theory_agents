"""Collects the written reasons of every game that involves the God description into one CSV file.

Usage (from the project folder):
    python export_god_reasons.py
    python export_god_reasons.py --part-a data/part_a_pd_christian_god_opponent --part-b data/part_b

Part A: both agents are told that they are devout Christians and that the other player is God;
        one row per player and round.
Part B: condition ALLC_GOD (the agent is told that God always cooperates; the opponent really does);
        one row per round (the agent only).

The file results/god_reasons.csv is meant for reading the reasons in Excel. It uses UTF-8 with a
byte order mark so that Excel opens it correctly.
"""

import argparse
import csv
import json
from pathlib import Path

PART_A_LABEL = "A: both agents told devout Christian and other player is God"
PART_B_LABEL = "B: agent told devout Christian and God always cooperates; opponent always cooperates"
COLUMNS = ["part", "condition", "stakes", "game", "game_length", "round", "player",
           "cooperative_letter", "choice_letter", "choice", "other_player_choice",
           "payoff", "reason"]


def read_games(folder):
    with open(Path(folder) / "games.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def kind(letter, coop_label):
    return "cooperate" if letter == coop_label else "betray"


def rows_for(game, part, label, players):
    coop = game["coop_label"]
    for number, rd in enumerate(game["rounds"], start=1):
        for index, name in players:
            other = 1 - index
            yield {
                "part": part, "condition": label, "stakes": game["stakes"],
                "game": game["game"], "game_length": game["n_rounds"], "round": number,
                "player": name, "cooperative_letter": coop,
                "choice_letter": rd["moves"][index],
                "choice": kind(rd["moves"][index], coop),
                "other_player_choice": kind(rd["moves"][other], coop),
                "payoff": rd["payoffs"][index], "reason": rd["reasons"][index],
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-a", default="data/part_a_pd_christian_god_opponent")
    parser.add_argument("--part-b", default="data/part_b")
    parser.add_argument("--out", default="results/god_reasons.csv")
    args = parser.parse_args()

    rows = []
    for game in read_games(args.part_a):
        if game.get("opponent_note") == "christian_god" and game.get("game_type", "pd") == "pd":
            rows.extend(rows_for(game, "A", PART_A_LABEL, [(0, 1), (1, 2)]))
    for game in read_games(args.part_b):
        if game.get("opponent") == "ALLC_GOD":
            rows.extend(rows_for(game, "B", PART_B_LABEL, [(0, "agent")]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows written to {out}")


if __name__ == "__main__":
    main()
