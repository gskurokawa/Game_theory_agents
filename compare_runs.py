"""Compares two runs of the study, game type by game type and stakes level by stakes level.

Usage (from the project folder):
    python compare_runs.py --a data/part_a --b data/part_a_ai_opponent

Run A is the reference (for example the first run). Every difference is B minus A.
The report is printed and written to results/compare_<A>_vs_<B>.txt.

For each game and stakes level it gives:
  - the average behaviour per game in A and in B, the difference, its 95%
    bootstrap confidence interval and a permutation test p-value;
  - the pooled counts of cooperative choices, and of cooperative choices in round 1;
  - for the whole run: how often the written reasons use certain arguments,
    including a check of whether the agents mention that the other player is
    an AI or the same model.
"""

import argparse
import re
from pathlib import Path

import numpy as np

import analyze_part_a as A
import settings
from run_part_a import read_jsonl

# Word patterns used to count arguments in the written reasons (approximate).
PATTERNS = {
    "one-round argument ('better whatever the other player does')":
        re.compile(r"regardless|whatever the other|no matter|dominant", re.I),
    "mentions future, repeated play or reciprocity":
        re.compile(r"future|repeated|many rounds|long.?term|sustain|reciproc|build trust|"
                   r"mutual(ly)? (beneficial|cooperation)|retaliat|punish|reputation|continu", re.I),
    "mentions that the other player is an AI or the same model":
        re.compile(r"\bAI\b|same model|same as (me|you)|identical|symmetric|mirror|also gpt|"
                   r"gpt-?5|another instance|copy of|another agent|like me|like you", re.I),
    "mentions the bonus, prize or threshold":
        re.compile(r"bonus|prize|threshold|average", re.I),
    "uses religious content (Christian, faith, moral, forgiveness; not the bare word God)":
        re.compile(r"Christian|faith|divine|religio|\bsin\b|scripture|golden rule|"
                   r"neighbou?r|moral|righteous|forgiv|devout|pray|bless|mercy|grace|\bLord\b|"
                   r"Bible|Christ", re.I),
    "uses the word God (also the name the agent gives to the other player)":
        re.compile(r"\bGod", re.I),
    "mentions philanthropy, generosity or charity":
        re.compile(r"philanthrop|generous|generosity|charit|altruis|benevolen", re.I),
}


def load(folder):
    records = read_jsonl(Path(folder) / "games.jsonl")
    if not records:
        raise SystemExit(f"No finished games in {folder}")
    return records


def reasons_of(records):
    return [t for g in records for rd in g["rounds"] for t in rd["reasons"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", required=True, help="data folder of run A (reference)")
    parser.add_argument("--b", required=True, help="data folder of run B")
    parser.add_argument("--out", default=None, help="output text file")
    args = parser.parse_args()

    rec_a, rec_b = load(args.a), load(args.b)
    games_a, rounds_a = A.build_tables(rec_a)
    games_b, rounds_b = A.build_tables(rec_b)
    rng = np.random.default_rng(settings.STUDY_SEED)

    lines = []
    out = lines.append
    out("COMPARISON OF TWO RUNS")
    out("=" * 60)
    out(f"A (reference): {args.a}   {len(rec_a)} games")
    out(f"B:             {args.b}   {len(rec_b)} games")
    out("All differences are B minus A.")
    out("Average per game: each game gives one value (prisoner's dilemma: share of "
        "choices that were the cooperative option; public goods game: contribution "
        "divided by the amount received). 95% confidence interval: bootstrap with "
        f"{A.N_RESAMPLES:,} resamples of the games. p-value: permutation test "
        f"with {A.N_RESAMPLES:,} random reassignments of the games to A and B.")
    out("Several comparisons are made below (one per game and stakes level), so one "
        "small p-value among them can arise by chance. A p-value below about 0.0125 "
        "is more convincing when four comparisons are made.")
    out("")

    for game_type in settings.GAME_TYPES:
        for stakes in ("low", "high"):
            ga = games_a[(games_a.game_type == game_type) & (games_a.stakes == stakes)]
            gb = games_b[(games_b.game_type == game_type) & (games_b.stakes == stakes)]
            if ga.empty or gb.empty:
                continue
            ra = rounds_a[(rounds_a.game_type == game_type) & (rounds_a.stakes == stakes)]
            rb = rounds_b[(rounds_b.game_type == game_type) & (rounds_b.stakes == stakes)]
            name = A.NAMES[game_type]
            out(f"{name.upper()}, {stakes.upper()} STAKES")
            out("-" * 60)
            diff, (lo, hi), p = A.compare(ga.level.values, gb.level.values, rng)
            out(f"Average per game: A {ga.level.mean():.3f} ({len(ga)} games), "
                f"B {gb.level.mean():.3f} ({len(gb)} games). Difference {diff:+.3f} "
                f"(95% confidence interval {lo:+.3f} to {hi:+.3f}); p-value {p:.4f}.")
            if game_type == "pd":
                for label, r in (("A", ra), ("B", rb)):
                    r1 = r[r["round"] == 1]
                    out(f"  {label}: cooperative choices {int(r.x.sum())} of {len(r)} "
                        f"({r.x.mean():.1%}); in round 1: {int(r1.x.sum())} of {len(r1)}; "
                        f"games with at least one cooperative choice: "
                        f"{int((r.groupby('game').x.sum() > 0).sum())} of {r.game.nunique()}.")
            else:
                for label, r in (("A", ra), ("B", rb)):
                    r1 = r[r["round"] == 1]
                    rounds_with = r.groupby(["game", "round"]).x.max()
                    out(f"  {label}: average contribution share {r.x.mean():.3f}; in round 1 "
                        f"{r1.x.mean():.3f}; rounds in which at least one player "
                        f"contributed: {int((rounds_with > 0).sum())} of {len(rounds_with)}.")
            out("")

    out("WRITTEN REASONS (share of all reasons; approximate, found by word patterns)")
    out("-" * 60)
    common = {g["game_type"] for g in rec_a} & {g["game_type"] for g in rec_b}
    out(f"Only games of the types present in both runs are counted: {sorted(common)}.")
    ta = reasons_of([g for g in rec_a if g["game_type"] in common])
    tb = reasons_of([g for g in rec_b if g["game_type"] in common])
    for label, rx in PATTERNS.items():
        ca = sum(1 for t in ta if rx.search(t))
        cb = sum(1 for t in tb if rx.search(t))
        out(f"{label}: A {ca} of {len(ta)} ({ca / len(ta):.1%}); B {cb} of {len(tb)} ({cb / len(tb):.1%}).")
    out("The third line is a check of the manipulation: if the agents were told who the "
        "other player is but almost never refer to it, the information may have had "
        "little influence.")

    report = "\n".join(lines)
    out_path = Path(args.out) if args.out else (
        settings.PROJECT_DIR / "results"
        / f"compare_{Path(args.a).name}_vs_{Path(args.b).name}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
