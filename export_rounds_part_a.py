"""Writes a readable table of Part A with one row per round.

Usage (from the project folder):
    python export_rounds_part_a.py
    python export_rounds_part_a.py --data data/part_a_pilot --out results/part_a_pilot_rounds.csv

The file opens in Excel. It uses only the standard library.

Columns
    condition, game_type, stakes   pd = prisoner's dilemma, pgg = public goods game
    game                           game number within the condition (from 0)
    game_rounds                    number of rounds the whole game lasted
    round                          round number
    p0_move, p1_move               what each player chose (PD: A or B; PGG: pounds contributed)
    cooperative_option             PD only: the option (A or B) that was cooperative in this game
    p0_cooperated, p1_cooperated   PD only: 1 if the player chose the cooperative option, else 0
    p0_share, p1_share             PGG only: contribution divided by the amount received
    p0_payoff, p1_payoff           money received in the round
    pool_after_multiplier          PGG only: the pool after multiplication
    p0_reason, p1_reason           the one-sentence reason each player gave
"""

import argparse
import csv
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ORDER = {"pd_low": 0, "pd_high": 1, "pgg_low": 2, "pgg_high": 3}
PGG_ENDOWMENT = 10  # at low stakes; multiplied by the game's scale


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=str(PROJECT_DIR / "data" / "part_a"))
    parser.add_argument("--out", default=None,
                        help="output file (default: results/part_a_rounds.csv for data/part_a, "
                             "otherwise results/<name of the data folder>/part_a_rounds.csv)")
    args = parser.parse_args()
    if args.out is None:
        data_name = Path(args.data).name
        folder = PROJECT_DIR / "results"
        if data_name != "part_a":
            folder = folder / data_name
        args.out = str(folder / "part_a_rounds.csv")

    path = Path(args.data) / "games.jsonl"
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    records.sort(key=lambda g: (ORDER.get(g["condition"], 9), g["game"]))

    columns = [
        "condition", "game_type", "stakes", "game", "game_rounds", "round",
        "p0_move", "p1_move", "cooperative_option", "p0_cooperated",
        "p1_cooperated", "p0_share", "p1_share", "p0_payoff", "p1_payoff",
        "pool_after_multiplier", "p0_reason", "p1_reason",
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    # utf-8 with a byte order mark, so that Excel shows the pound sign correctly
    with open(out_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for g in records:
            endowment = PGG_ENDOWMENT * g["scale"]
            for rec in g["rounds"]:
                row = {
                    "condition": g["condition"],
                    "game_type": g["game_type"],
                    "stakes": g["stakes"],
                    "game": g["game"],
                    "game_rounds": g["n_rounds"],
                    "round": rec["round"],
                    "p0_move": rec["moves"][0],
                    "p1_move": rec["moves"][1],
                    "p0_payoff": rec["payoffs"][0],
                    "p1_payoff": rec["payoffs"][1],
                    "p0_reason": rec["reasons"][0],
                    "p1_reason": rec["reasons"][1],
                }
                if g["game_type"] == "pd":
                    row["cooperative_option"] = g["coop_label"]
                    row["p0_cooperated"] = int(rec["moves"][0] == g["coop_label"])
                    row["p1_cooperated"] = int(rec["moves"][1] == g["coop_label"])
                else:
                    row["p0_share"] = round(rec["moves"][0] / endowment, 4)
                    row["p1_share"] = round(rec["moves"][1] / endowment, 4)
                    row["pool_after_multiplier"] = rec["pool"]
                writer.writerow(row)
                rows += 1
    print(f"{len(records)} games, {rows} rounds written to {out_path}")


if __name__ == "__main__":
    main()
