"""Analysis of Part B: one AI agent against fixed strategies.

Usage (from the project folder):
    python analyze_part_b.py                          # reads data/part_b
    python analyze_part_b.py --data data/part_b_mock  # another folder

Writes a text report and a chart to results/part_b/ (or results/<folder name>/).

The unit of analysis is the game. For each game the program computes the share of the
AI agent's choices that were cooperative, its average money per round, and what a few
simple fixed strategies would have earned against the same opponent and the same game
length (the benchmark).
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_part_a as A
import games
import settings
import strategies
from run_part_a import read_jsonl

BLOCKS = [("rounds 1-3", 1, 3), ("rounds 4-7", 4, 7), ("rounds 8-12", 8, 12), ("rounds 13+", 13, 99)]
BENCHMARK_AGENTS = {
    "always betray": strategies.always_betray,
    "tit-for-tat": strategies.tit_for_tat,
    "always cooperate": strategies.always_cooperate,
    "grim trigger": strategies.grim_trigger,
}
PATTERNS = {
    "says the other player always or consistently makes the same choice":
        re.compile(r"\balways\b|consistent|every round|each round|keeps? (choosing|playing)", re.I),
    "says the other player copies or mirrors the agent's choices":
        re.compile(r"mirror|copies|copying|tit.for.tat|imitat|retaliat|reciprocat|responds? to my", re.I),
    "says the other player may be random":
        re.compile(r"random|unpredictab|no pattern", re.I),
    "uses religious content (Christian, faith, moral, forgiveness, not the word God)":
        re.compile(r"Christian|faith|divine|religio|\bsin\b|scripture|golden rule|neighbou?r|"
                   r"moral|righteous|forgiv|devout|pray|bless|mercy|grace|\bLord\b|Bible|Christ",
                   re.I),
    "mentions philanthropy, generosity or charity":
        re.compile(r"philanthrop|generous|generosity|charit|altruis|benevolen", re.I),
    "mentions that exploiting the other player is possible or profitable":
        re.compile(r"exploit|take advantage|always cooperat|keeps? cooperat|guarantee.*(5|£5)|"
                   r"£5 (rather|instead)|higher payoff", re.I),
    "mentions the future, repeated play or building trust":
        re.compile(r"future|repeated|long.?term|build trust|sustain|mutual", re.I),
}


def benchmark_total(agent_fn, opponent, index, length, scale):
    """Total money of a fixed strategy against the same opponent and game length."""
    rng = strategies.game_rng(settings.STUDY_SEED, opponent, index)
    opp_fn = strategies.OPPONENTS[opponent]
    mine, theirs = [], []
    total = 0.0
    for _ in range(length):
        a = agent_fn(mine, theirs, None)
        o = opp_fn(theirs, mine, rng)
        total += settings.PD_PAYOFFS[(a, o)] * scale
        mine.append(a)
        theirs.append(o)
    return total


def build_tables(records):
    game_rows, round_rows = [], []
    for g in records:
        scale = g["scale"]
        types = [r["agent_type"] for r in g["rounds"]]
        coop = np.array([t == "C" for t in types], dtype=float)
        total = g["total_payoffs"][0]
        row = {
            "opponent": g["opponent"], "stakes": g["stakes"], "game": g["game"],
            "n_rounds": g["n_rounds"], "level": coop.mean(), "first_round": coop[0],
            "total": total, "per_round": total / g["n_rounds"] / scale,
            "best_total": max(benchmark_total(f, g["opponent"], g["game"], g["n_rounds"], scale)
                              for f in BENCHMARK_AGENTS.values()),
        }
        for name, fn in BENCHMARK_AGENTS.items():
            row["bm_" + name] = benchmark_total(fn, g["opponent"], g["game"], g["n_rounds"], scale)
        game_rows.append(row)
        for r in g["rounds"]:
            prev = None if r["round"] == 1 else g["rounds"][r["round"] - 2]["opponent_type"]
            round_rows.append({
                "opponent": g["opponent"], "stakes": g["stakes"], "game": g["game"],
                "round": r["round"], "x": float(r["agent_type"] == "C"),
                "prev_opponent": prev,
            })
    return pd.DataFrame(game_rows), pd.DataFrame(round_rows)


def draw_chart(rounds, path, stakes):
    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    palette = {"TFT": "#2a78d6", "ALLD": "#eb6834", "ALLC": "#2a9d6f", "GRIM": "#8a5cd0",
               "RANDOM": "#8a8985", "ALLC_GOD": "#d6a82a", "ALLC_TOLD": "#c23f7a",
               "ALLC_PHIL": "#2aa5b8", "ALLD_PHIL": "#7a4a2a"}
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=surface)
    ax.set_facecolor(surface)
    sub = rounds[rounds.stakes == stakes]
    reach = sub.groupby(["opponent", "round"])["game"].nunique().unstack(0)
    last = int(reach[(reach >= 5).all(axis=1)].index.max()) if (reach >= 5).all(axis=1).any() else 8
    for name in strategies.OPPONENTS:
        s = sub[(sub.opponent == name) & (sub["round"] <= last)]
        if s.empty:
            continue
        by_round = s.groupby("round")["x"].mean()
        ax.plot(by_round.index, by_round.values, color=palette[name], linewidth=2, marker="o",
                markersize=5, markeredgecolor=surface, markeredgewidth=1.2, label=name)
    ax.set_ylim(0, 1)
    ax.set_xlim(0.7, last + 0.3)
    ax.set_xticks(range(1, last + 1))
    ax.set_xlabel("Round", color=ink2)
    ax.set_ylabel("Share of games in which the AI agent cooperated", color=ink2)
    ax.set_title(f"Cooperation of the AI agent by round and opponent ({stakes} stakes)",
                 color=ink, fontsize=12, loc="left")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data/part_b", help="data folder")
    args = parser.parse_args()
    data_dir = Path(args.data)
    records = read_jsonl(data_dir / "games.jsonl")
    if not records:
        raise SystemExit(f"No finished games in {data_dir}")
    calls = read_jsonl(data_dir / "calls.jsonl")
    failures = read_jsonl(data_dir / "failures.jsonl")
    games_df, rounds_df = build_tables(records)
    rng = np.random.default_rng(settings.STUDY_SEED)
    names = [n for n in strategies.OPPONENTS if (games_df.opponent == n).any()]

    lines = []
    out = lines.append
    out("PART B: ONE AI AGENT AGAINST FIXED STRATEGIES")
    out("=" * 60)
    out(f"Data folder: {data_dir}")
    out(f"Model: {sorted({r['model'] for r in records})}")
    out("The AI agent was told only that there is one other player. It was not told "
        "that the other player is a fixed strategy, nor which one (except in the conditions "
        "ALLC_GOD, ALLC_TOLD, ALLC_PHIL and ALLD_PHIL, which are described below). Payoffs: both "
        "cooperate 3; cooperate while the other betrays 0; betray while the other "
        "cooperates 5; both betray 1 (pounds per round at low stakes).")
    out("The game ends with probability 10% after each round and always after round "
        f"{settings.MAX_ROUNDS}; the agent was told both rules.")
    out(f"Games analysed: {len(games_df)}. Failure log entries: {len(failures)}.")
    if calls:
        invalid = sum(1 for c in calls if not c["valid"])
        out(f"Answers that could not be read: {invalid} of {len(calls)} calls "
            f"({invalid / len(calls):.1%}); each was asked again, up to "
            f"{settings.MAX_PARSE_ATTEMPTS} times.")
    out("")
    out("OPPONENTS")
    for n in names:
        out(f"  {n}: {strategies.DESCRIPTIONS[n]}")
    out("")
    out("Conditions with a sentence for the agent. In ALLC_GOD, ALLC_TOLD and ALLC_PHIL the "
        "opponent is the same as in ALLC (it always cooperates). ALLC_GOD: 'You are a devout "
        "Christian. The other player is God, and God always cooperates.' ALLC_TOLD (control): "
        "'The other player always cooperates.' ALLC_PHIL: 'The other player is one of the "
        "world's greatest philanthropists.' In ALLD_PHIL the same philanthropist sentence is "
        "given, but the opponent always betrays, so the description is misleading. The "
        "religious framing and the philanthropist description are not literally true; the "
        "statement that the other player always cooperates is true.") if any(
        n in names for n in ("ALLC_GOD", "ALLC_TOLD", "ALLC_PHIL", "ALLD_PHIL")) else None
    out("DEFINITIONS")
    out("Cooperation rate: the share of the AI agent's choices in a game that were the "
        "cooperative option. Each game gives one value; the tables average games, so every "
        "game has the same weight whatever its length.")
    out("Money per round: the AI agent's total money in a game divided by the number of "
        "rounds (at low stakes; at other stakes it is divided by the stakes factor too).")
    out("Benchmark: the money that a fixed strategy (always betray, tit-for-tat, always "
        "cooperate or grim trigger) would have earned against the same opponent in a game of "
        "the same length. 'Best of the four' is the highest of these totals in each game. "
        "For the random opponent the benchmark uses the same random choices as the real game.")
    out("95% confidence interval: percentile bootstrap with "
        f"{A.N_RESAMPLES:,} resamples of the games. p-value: permutation test with "
        f"{A.N_RESAMPLES:,} reassignments of the games to the two groups.")
    out("")

    for stakes in sorted(games_df.stakes.unique(), key=lambda s: settings.STAKES[s]):
        gs = games_df[games_df.stakes == stakes]
        rs = rounds_df[rounds_df.stakes == stakes]
        out(f"RESULTS, {stakes.upper()} STAKES")
        out("-" * 60)
        out(f"{'opponent':<8}{'games':>6}{'mean len':>9}{'coop rate (95% CI)':>26}"
            f"{'round 1':>9}{'money/round':>13}{'best of 4':>11}")
        for n in names:
            g = gs[gs.opponent == n]
            if g.empty:
                continue
            lo, hi = A.bootstrap_ci(g.level, rng)
            best_pr = (g.best_total / g.n_rounds / settings.STAKES[stakes]).mean()
            out(f"{n:<8}{len(g):>6}{g.n_rounds.mean():>9.1f}"
                f"{g.level.mean():>11.3f} ({lo:.3f}-{hi:.3f})"
                f"{g.first_round.mean():>9.2f}{g.per_round.mean():>13.2f}{best_pr:>11.2f}")
        out("")
        out("Money per round of each benchmark strategy against each opponent:")
        out(f"{'opponent':<8}" + "".join(f"{k:>18}" for k in BENCHMARK_AGENTS))
        for n in names:
            g = gs[gs.opponent == n]
            if g.empty:
                continue
            cells = "".join(
                f"{(g['bm_' + k] / g.n_rounds / settings.STAKES[stakes]).mean():>18.2f}"
                for k in BENCHMARK_AGENTS)
            out(f"{n:<8}{cells}")
        out("")

        out("Cooperation rate of the AI agent by stage of the game (share of choices):")
        out(f"{'opponent':<8}" + "".join(f"{label:>14}" for label, _, _ in BLOCKS))
        for n in names:
            r = rs[rs.opponent == n]
            cells = ""
            for _, a, b in BLOCKS:
                sel = r[(r["round"] >= a) & (r["round"] <= b)]
                cells += f"{sel.x.mean():>10.3f}({len(sel):>3})" if len(sel) else f"{'-':>14}"
            out(f"{n:<8}{cells}")
        out("(the number in brackets is the number of choices)")
        out("")

        out("Reaction to the opponent's previous choice (rounds 2 onwards, pooled; rounds "
            "within a game are not independent):")
        for n in names:
            r = rs[(rs.opponent == n) & rs.prev_opponent.notna()]
            if r.empty:
                continue
            after_c = r[r.prev_opponent == "C"].x
            after_d = r[r.prev_opponent == "D"].x
            f = lambda s: f"{s.mean():.3f} (n={len(s)})" if len(s) else "no cases"
            out(f"  {n:<7} cooperation after the opponent cooperated: {f(after_c)}; "
                f"after it betrayed: {f(after_d)}")
        out("")

        out("COMPARISONS OF COOPERATION RATE BETWEEN OPPONENTS (difference of game averages)")
        out("Several comparisons are made, so a p-value below about 0.0125 is more convincing.")
        pairs = [("TFT", "ALLD"), ("GRIM", "ALLD"), ("ALLC", "ALLD"), ("TFT", "ALLC"),
                 ("TFT", "RANDOM"), ("ALLC_TOLD", "ALLC"), ("ALLC_GOD", "ALLC_TOLD"),
                 ("ALLC_GOD", "ALLC"), ("ALLC_PHIL", "ALLC_TOLD"), ("ALLC_PHIL", "ALLC_GOD"),
                 ("ALLC_PHIL", "ALLC"), ("ALLD_PHIL", "ALLD")]
        for x, y in pairs:
            gx, gy = gs[gs.opponent == x].level.values, gs[gs.opponent == y].level.values
            if len(gx) < 2 or len(gy) < 2:
                continue
            diff, (lo, hi), p = A.compare(gy, gx, rng)
            out(f"  {x} minus {y}: {diff:+.3f} (95% confidence interval {lo:+.3f} to "
                f"{hi:+.3f}); p-value {p:.4f}")
        out("")

        out("Money earned against the best of the four benchmark strategies "
            "(per round, game average):")
        for n in names:
            g = gs[gs.opponent == n]
            if g.empty:
                continue
            gap = (g.best_total - g.total) / g.n_rounds / settings.STAKES[stakes]
            lo, hi = A.bootstrap_ci(gap, rng)
            out(f"  {n:<7} shortfall {gap.mean():.2f} per round (95% confidence interval "
                f"{lo:.2f} to {hi:.2f}); the agent matched or beat the best benchmark in "
                f"{int((g.total >= g.best_total - 1e-9).sum())} of {len(g)} games")
        out("")

    reasons = {n: [t for g in records if g["opponent"] == n for r in g["rounds"]
                   for t in [r["reasons"][0]]] for n in names}
    out("WRITTEN REASONS (share of the agent's reasons; approximate, found by word patterns)")
    out("-" * 60)
    out(f"{'':<64}" + "".join(f"{n:>11}" for n in names))
    for label, rx in PATTERNS.items():
        cells = "".join(
            f"{sum(1 for t in reasons[n] if rx.search(t)) / max(1, len(reasons[n])):>11.0%}"
            for n in names)
        out(f"{label[:62]:<64}{cells}")
    out("")

    report = "\n".join(lines)
    results_dir = settings.PROJECT_DIR / "results" / (
        "part_b" if data_dir.name == "part_b" else data_dir.name)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "part_b_report.txt").write_text(report + "\n", encoding="utf-8")
    for stakes in games_df.stakes.unique():
        draw_chart(rounds_df, results_dir / f"part_b_by_round_{stakes}.png", stakes)
    games_df.drop(columns=["best_total"]).to_csv(results_dir / "part_b_games.csv", index=False)
    print(report)
    print(f"\nFiles written to {results_dir}")


if __name__ == "__main__":
    main()
