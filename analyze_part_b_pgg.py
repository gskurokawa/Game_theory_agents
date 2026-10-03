"""Analysis of Part B, public goods version: one AI agent against fixed strategies.

Usage (from the project folder):
    python analyze_part_b_pgg.py                                   # reads data/part_b_pgg
    python analyze_part_b_pgg.py --data data/part_b_pgg_gpt6luna   # another folder

Writes a text report, a chart and a table of games to results/<folder name>/.

The unit of analysis is the game. For each game the program computes the contribution
share of the AI agent (its contribution divided by the endowment, averaged over the rounds
of the game), its average money per round, and what a few simple fixed strategies would
have earned against the same opponent and the same game length (the benchmark).
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
    "contribute nothing": strategies.pgg_contribute_nothing,
    "copy the other": strategies.pgg_copy,
    "contribute everything": strategies.pgg_contribute_everything,
    "grim trigger": strategies.pgg_grim,
}
PATTERNS = {
    "says the other player always contributes the same amount (everything or nothing)":
        re.compile(r"\balways\b|consistent|every round|each round|keeps? (contributing|giving)", re.I),
    "says the other player copies or mirrors the agent's contributions":
        re.compile(r"mirror|copies|copying|tit.for.tat|imitat|retaliat|reciprocat|matches? my", re.I),
    "says the other player may be random":
        re.compile(r"random|unpredictab|no pattern", re.I),
    "mentions keeping the money, free riding or contributing nothing being best":
        re.compile(r"free.?rid|keep(ing)? (it|the|my|all)|contribut\w* (nothing|0|zero)|"
                   r"\b0 is\b|dominant|each pound|£0\.8|0\.8", re.I),
    "mentions the pool, the multiplier or joint gains":
        re.compile(r"pool|multipl|1\.6|joint|mutual|both (players )?(gain|benefit|earn)", re.I),
    "uses religious content (Christian, faith, moral, forgiveness, not the word God)":
        re.compile(r"Christian|faith|divine|religio|\bsin\b|scripture|golden rule|neighbou?r|"
                   r"moral|righteous|forgiv|devout|pray|bless|mercy|grace|\bLord\b|Bible|Christ",
                   re.I),
    "mentions philanthropy, generosity or charity":
        re.compile(r"philanthrop|generous|generosity|charit|altruis|benevolen", re.I),
    "mentions the future, repeated play or building trust":
        re.compile(r"future|repeated|long.?term|build trust|sustain", re.I),
}


def benchmark_total(agent_fn, opponent, index, length, scale):
    """Total money of a fixed strategy against the same opponent and game length."""
    endowment = settings.PGG_ENDOWMENT * scale
    rng = strategies.game_rng(settings.STUDY_SEED, opponent, index)
    opp_fn = strategies.PGG_OPPONENTS[opponent]
    mine, theirs = [], []
    total = 0.0
    for _ in range(length):
        a = agent_fn(mine, theirs, None, endowment)
        o = opp_fn(theirs, mine, rng, endowment)
        total += games.pgg_round([a, o], scale)[0][0]
        mine.append(a)
        theirs.append(o)
    return total


def build_tables(records):
    game_rows, round_rows = [], []
    for g in records:
        scale = g["scale"]
        share = np.array([r["agent_share"] for r in g["rounds"]], dtype=float)
        total = g["total_payoffs"][0]
        bm = {name: benchmark_total(fn, g["opponent"], g["game"], g["n_rounds"], scale)
              for name, fn in BENCHMARK_AGENTS.items()}
        row = {
            "opponent": g["opponent"], "stakes": g["stakes"], "game": g["game"],
            "n_rounds": g["n_rounds"], "level": share.mean(), "first_round": share[0],
            "zero_rounds": float((share == 0).mean()),
            "total": total, "per_round": total / g["n_rounds"] / scale,
            "best_total": max(bm.values()),
        }
        row.update({"bm_" + k: v for k, v in bm.items()})
        game_rows.append(row)
        for r in g["rounds"]:
            prev = None if r["round"] == 1 else g["rounds"][r["round"] - 2]["opponent_share"]
            round_rows.append({
                "opponent": g["opponent"], "stakes": g["stakes"], "game": g["game"],
                "round": r["round"], "x": r["agent_share"], "prev_opponent": prev,
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
    for name in strategies.PGG_OPPONENTS:
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
    ax.set_ylabel("Average contribution share of the AI agent", color=ink2)
    ax.set_title(f"Contribution of the AI agent by round and opponent ({stakes} stakes)",
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
    parser.add_argument("--data", default="data/part_b_pgg", help="data folder")
    args = parser.parse_args()
    data_dir = Path(args.data)
    records = read_jsonl(data_dir / "games.jsonl")
    if not records:
        raise SystemExit(f"No finished games in {data_dir}")
    calls = read_jsonl(data_dir / "calls.jsonl")
    failures = read_jsonl(data_dir / "failures.jsonl")
    games_df, rounds_df = build_tables(records)
    rng = np.random.default_rng(settings.STUDY_SEED)
    names = [n for n in strategies.PGG_OPPONENTS if (games_df.opponent == n).any()]

    lines = []
    out = lines.append
    out("PART B, PUBLIC GOODS GAME: ONE AI AGENT AGAINST FIXED STRATEGIES")
    out("=" * 60)
    out(f"Data folder: {data_dir}")
    out(f"Model: {sorted({r['model'] for r in records})}")
    out("The AI agent was told only that there is one other player. It was not told that the "
        "other player is a fixed strategy, nor which one (except in the conditions ALLC_GOD, "
        "ALLC_TOLD, ALLC_PHIL and ALLD_PHIL, described below). Each round both players receive "
        f"{settings.PGG_ENDOWMENT} pounds (at low stakes) and choose how much to put into a "
        f"shared pool; the pool is multiplied by {settings.PGG_MULTIPLIER:g} and divided equally.")
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
        out(f"  {n}: {strategies.PGG_DESCRIPTIONS[n]}")
    out("")
    out("DEFINITIONS")
    out("Contribution share: the agent's contribution in a round divided by the amount it "
        "received (0 means nothing, 1 means everything), averaged over the rounds of a game. "
        "Each game gives one value; the tables average games, so every game has the same "
        "weight whatever its length.")
    out("Zero rounds: the share of the rounds of a game in which the agent contributed nothing.")
    out("Money per round: the agent's total money in a game divided by the number of rounds "
        "(at low stakes; at other stakes it is divided by the stakes factor too). A player "
        f"that contributes nothing and meets a full contributor receives "
        f"{settings.PGG_ENDOWMENT * (1 + settings.PGG_MULTIPLIER / 2):g}; two full contributors "
        f"receive {settings.PGG_ENDOWMENT * settings.PGG_MULTIPLIER:g} each; two players that "
        f"contribute nothing receive {settings.PGG_ENDOWMENT:g} each.")
    out("Benchmark: the money that a fixed strategy (contribute nothing, copy the other "
        "player, contribute everything, or grim trigger) would have earned against the same "
        "opponent in a game of the same length. 'Best of the four' is the highest of these "
        "totals in each game. For the random opponent the benchmark uses the same random "
        "contributions as the real game.")
    out("95% confidence interval: percentile bootstrap with "
        f"{A.N_RESAMPLES:,} resamples of the games. p-value: permutation test with "
        f"{A.N_RESAMPLES:,} reassignments of the games to the two groups.")
    out("")

    for stakes in sorted(games_df.stakes.unique(), key=lambda s: settings.STAKES[s]):
        gs = games_df[games_df.stakes == stakes]
        rs = rounds_df[rounds_df.stakes == stakes]
        out(f"RESULTS, {stakes.upper()} STAKES")
        out("-" * 60)
        out(f"{'opponent':<10}{'games':>6}{'mean len':>9}{'contribution (95% CI)':>29}"
            f"{'round 1':>9}{'zero rds':>10}{'money/rd':>10}{'best of 4':>10}")
        for n in names:
            g = gs[gs.opponent == n]
            if g.empty:
                continue
            lo, hi = A.bootstrap_ci(g.level, rng)
            best_pr = (g.best_total / g.n_rounds / settings.STAKES[stakes]).mean()
            out(f"{n:<10}{len(g):>6}{g.n_rounds.mean():>9.1f}"
                f"{g.level.mean():>14.3f} ({lo:.3f}-{hi:.3f})"
                f"{g.first_round.mean():>9.2f}{g.zero_rounds.mean():>10.2f}"
                f"{g.per_round.mean():>10.2f}{best_pr:>10.2f}")
        out("")
        out("Money per round of each benchmark strategy against each opponent:")
        out(f"{'opponent':<10}" + "".join(f"{k:>23}" for k in BENCHMARK_AGENTS))
        for n in names:
            g = gs[gs.opponent == n]
            if g.empty:
                continue
            cells = "".join(
                f"{(g['bm_' + k] / g.n_rounds / settings.STAKES[stakes]).mean():>23.2f}"
                for k in BENCHMARK_AGENTS)
            out(f"{n:<10}{cells}")
        out("")

        out("Contribution share of the AI agent by stage of the game (average over choices):")
        out(f"{'opponent':<10}" + "".join(f"{label:>14}" for label, _, _ in BLOCKS))
        for n in names:
            r = rs[rs.opponent == n]
            cells = ""
            for _, a, b in BLOCKS:
                sel = r[(r["round"] >= a) & (r["round"] <= b)]
                cells += f"{sel.x.mean():>10.3f}({len(sel):>3})" if len(sel) else f"{'-':>14}"
            out(f"{n:<10}{cells}")
        out("(the number in brackets is the number of choices)")
        out("")

        out("Reaction to the opponent's previous contribution (rounds 2 onwards, pooled; "
            "rounds within a game are not independent):")
        for n in names:
            r = rs[(rs.opponent == n) & rs.prev_opponent.notna()]
            if r.empty:
                continue
            high = r[r.prev_opponent >= 0.5].x
            low = r[r.prev_opponent < 0.5].x
            f = lambda s: f"{s.mean():.3f} (n={len(s)})" if len(s) else "no cases"
            out(f"  {n:<9} contribution share after the opponent contributed at least half: "
                f"{f(high)}; after it contributed less than half: {f(low)}")
        out("")

        out("COMPARISONS OF CONTRIBUTION SHARE BETWEEN OPPONENTS (difference of game averages)")
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
            out(f"  {n:<9} shortfall {gap.mean():.2f} per round (95% confidence interval "
                f"{lo:.2f} to {hi:.2f}); the agent matched or beat the best benchmark in "
                f"{int((g.total >= g.best_total - 1e-9).sum())} of {len(g)} games")
        out("")

    reasons = {n: [r["reasons"][0] for g in records if g["opponent"] == n for r in g["rounds"]]
               for n in names}
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
    results_dir = settings.PROJECT_DIR / "results" / data_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "part_b_pgg_report.txt").write_text(report + "\n", encoding="utf-8")
    for stakes in games_df.stakes.unique():
        draw_chart(rounds_df, results_dir / f"part_b_pgg_by_round_{stakes}.png", stakes)
    games_df.drop(columns=["best_total"]).to_csv(results_dir / "part_b_pgg_games.csv", index=False)
    print(report)
    print(f"\nFiles written to {results_dir}")


if __name__ == "__main__":
    main()
