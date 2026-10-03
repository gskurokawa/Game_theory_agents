"""Analysis of Part A. Reads the data written by run_part_a.py.

Usage (from the project folder):
    python analyze_part_a.py                    analyses data/part_a
    python analyze_part_a.py --data data/part_a_mock

Writes three files into the folder results/ (or --results):
    part_a_report.txt      the numbers, with definitions
    part_a_summary.csv     one row per condition
    part_a_by_round.png    average behaviour in each round, low against high stakes

The unit of analysis is the GAME, not the round, because the rounds inside one
game depend on each other. Each game contributes one number to each average.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import settings
from run_part_a import read_jsonl

N_RESAMPLES = 10_000
MIN_GAMES_PER_ROUND = 10  # a round is drawn in the chart only if this many games reached it

NAMES = {"pd": "Prisoner's dilemma", "pgg": "Public goods game"}
METRIC = {
    "pd": "cooperation rate (share of choices that were the cooperative option)",
    "pgg": "contribution share (contribution divided by the amount received)",
}


# ------------------------------------------------------------- loading ----
def build_tables(game_records):
    """Returns (one row per game, one row per player and round)."""
    game_rows, round_rows = [], []
    for g in game_records:
        endowment = settings.PGG_ENDOWMENT * g["scale"]
        xs = []  # per round: [value of player 0, value of player 1]
        for rec in g["rounds"]:
            if g["game_type"] == "pd":
                xs.append([1.0 if m == g["coop_label"] else 0.0 for m in rec["moves"]])
            else:
                xs.append([m / endowment for m in rec["moves"]])
        for r, pair in enumerate(xs):
            for p in (0, 1):
                round_rows.append(
                    {
                        "condition": g["condition"],
                        "game_type": g["game_type"],
                        "stakes": g["stakes"],
                        "game": g["game"],
                        "round": r + 1,
                        "player": p,
                        "x": pair[p],
                        "other_prev": xs[r - 1][1 - p] if r > 0 else np.nan,
                    }
                )
        arr = np.array(xs)
        row = {
            "condition": g["condition"],
            "game_type": g["game_type"],
            "stakes": g["stakes"],
            "game": g["game"],
            "n_rounds": g["n_rounds"],
            "level": arr.mean(),
            "first_round": arr[0].mean(),
            "total_payoff_mean": float(np.mean(g["total_payoffs"])),
        }
        if g["game_type"] == "pd":
            row["mutual_cooperation"] = float((arr == 1.0).all(axis=1).mean())
        game_rows.append(row)
    return pd.DataFrame(game_rows), pd.DataFrame(round_rows)


# --------------------------------------------------------- statistics ----
def bootstrap_ci(values, rng):
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), (N_RESAMPLES, len(v)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def compare(low, high, rng):
    """Difference of the averages (high minus low), its 95% bootstrap interval,
    and a permutation test p-value."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    observed = high.mean() - low.mean()
    boot = (
        high[rng.integers(0, len(high), (N_RESAMPLES, len(high)))].mean(axis=1)
        - low[rng.integers(0, len(low), (N_RESAMPLES, len(low)))].mean(axis=1)
    )
    pooled = np.concatenate([low, high])
    extreme = 0
    for _ in range(N_RESAMPLES):
        rng.shuffle(pooled)
        diff = pooled[len(low):].mean() - pooled[: len(low)].mean()
        extreme += abs(diff) >= abs(observed) - 1e-12
    p_value = (extreme + 1) / (N_RESAMPLES + 1)
    return observed, np.percentile(boot, [2.5, 97.5]), p_value


# --------------------------------------------------------------- chart ----
def draw_chart(rounds, path):
    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    colours = {"low": "#2a78d6", "high": "#eb6834"}  # categorical slots 1 and 2
    labels = {"low": "Low stakes", "high": "High stakes (x 1,000)"}
    present = [g for g in settings.GAME_TYPES if (rounds.game_type == g).any()]
    fig, axes = plt.subplots(1, len(present), figsize=(5.5 * len(present) + 0.5, 4.4),
                             facecolor=surface, squeeze=False)
    for ax, game_type in zip(axes[0], present):
        ax.set_facecolor(surface)
        sub = rounds[rounds.game_type == game_type]
        reach = sub.groupby(["stakes", "round"])["game"].nunique().unstack(0)
        ok = reach[(reach >= MIN_GAMES_PER_ROUND).all(axis=1)]
        last = int(ok.index.max()) if len(ok) else int(sub["round"].max())
        for stakes in ("low", "high"):
            s = sub[(sub.stakes == stakes) & (sub["round"] <= last)]
            by_round = s.groupby("round")["x"].mean()
            ax.plot(by_round.index, by_round.values, color=colours[stakes],
                    linewidth=2, marker="o", markersize=6,
                    markeredgecolor=surface, markeredgewidth=1.5,
                    label=labels[stakes])
        ax.set_ylim(0, 1)
        ax.set_xlim(0.7, last + 0.3)
        ax.set_xticks(range(1, last + 1))
        ax.set_title(NAMES[game_type], color=ink, fontsize=12, loc="left")
        ax.set_xlabel("Round", color=ink2)
        ax.set_ylabel("Cooperation rate" if game_type == "pd"
                      else "Contribution share", color=ink2)
        ax.tick_params(colors=ink2)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#bdbcb6")
        ax.legend(frameon=False, labelcolor=ink2, loc="best")
    fig.text(0.01, 0.01,
             f"Average over games, both players. A round is drawn only while at "
             f"least {MIN_GAMES_PER_ROUND} games in each stakes level reached it.",
             color=ink2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


# ---------------------------------------------------------------- main ----
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=str(settings.PROJECT_DIR / "data" / "part_a"))
    parser.add_argument("--results", default=None,
                        help="output folder (default: results/ for data/part_a, "
                             "otherwise results/<name of the data folder>)")
    args = parser.parse_args()
    data_dir = Path(args.data)
    if args.results:
        out_dir = Path(args.results)
    elif data_dir.name == "part_a":
        out_dir = settings.PROJECT_DIR / "results"
    else:
        out_dir = settings.PROJECT_DIR / "results" / data_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(data_dir / "games.jsonl")
    calls = read_jsonl(data_dir / "calls.jsonl")
    failures = read_jsonl(data_dir / "failures.jsonl")
    if not records:
        raise SystemExit(f"No finished games in {data_dir}. Run run_part_a.py first.")
    games_df, rounds_df = build_tables(records)
    rng = np.random.default_rng(settings.STUDY_SEED)

    lines = []
    out = lines.append
    out("PART A: TWO AI AGENTS PLAYING EACH OTHER")
    out("=" * 60)
    out(f"Data folder: {data_dir}")
    out(f"Model: {sorted({r['model'] for r in records})}")
    notes = {("same_model" if r.get("opponent_note") is True else r.get("opponent_note"))
             for r in records if r.get("opponent_note")}
    if "christian_god" in notes:
        out("Each agent was told that it is a devout Christian and that the other player "
            "is God. Both statements are false: the other player is the same model as "
            "itself, and both agents received the same text.")
    elif "philanthropist" in notes:
        out("Each agent was told that the other player is one of the world's greatest "
            "philanthropists. This statement is false: the other player is the same "
            "model as itself.")
    elif notes:
        out("Each agent was told that the other player is an AI agent of the same "
            "model as itself.")
    else:
        out("The agents were not told who the other player is.")
    pd_records = [r for r in records if r["game_type"] == "pd"]
    if pd_records:
        variant = pd_records[0].get("pd_variant") or "standard"
        pay = pd_records[0].get("pd_payoffs") or {
            f"{m}{o}": v for (m, o), v in settings.PD_VARIANTS["standard"].items()
        }
        out(f"Prisoner's dilemma payoffs, version '{variant}' (pounds per round at low "
            f"stakes): both cooperate {pay['CC']}; you cooperate and the other player "
            f"betrays {pay['CD']}; you betray and the other player cooperates "
            f"{pay['DC']}; both betray {pay['DD']}.")
    prize_games = [r for r in pd_records if r.get("prize")]
    if prize_games:
        rule = prize_games[0]["prize"]
        out(f"Threshold bonus (prisoner's dilemma): each player received {rule['bonus']} "
            f"pounds at the end of the game if its average money per round over the "
            f"whole game was at least {rule['average']} pounds (at low stakes; both "
            f"amounts times 1,000 at high stakes). The agents were told this rule and "
            f"saw their running average each round.")
        for stakes in ("low", "high"):
            flags = [b for r in prize_games if r["stakes"] == stakes
                     for b in r["bonus_earned"]]
            if flags:
                out(f"  {stakes.capitalize()} stakes: bonus earned by {sum(flags)} of "
                    f"{len(flags)} players ({sum(flags) / len(flags):.0%}).")
    out(f"Games analysed: {len(games_df)} (planned: "
        f"{settings.N_GAMES * games_df.condition.nunique()}). "
        f"Failure log entries: {len(failures)} (a failed game is played again when "
        f"run_part_a.py is started again).")
    for cond, n in games_df.groupby("condition").size().items():
        out(f"  {cond}: {n} games")
    if calls:
        invalid = sum(1 for c in calls if not c["valid"])
        out(f"Answers that could not be read: {invalid} of {len(calls)} calls "
            f"({invalid / len(calls):.1%}); each was asked again, up to "
            f"{settings.MAX_PARSE_ATTEMPTS} times.")
    out("")
    out("DEFINITIONS")
    out("Cooperation rate (prisoner's dilemma): the share of all choices, by both "
        "players in all rounds of a game, that were the cooperative option.")
    out("Contribution share (public goods game): the amount a player put into the "
        "shared pool divided by the amount the player received in that round "
        "(0 means nothing, 1 means everything), averaged over players and rounds.")
    out("Each game gives one value. The tables below average the values of games, "
        "so every game has the same weight whatever its length.")
    out("95% confidence interval: the range obtained by resampling the games "
        f"{N_RESAMPLES:,} times with replacement and keeping the middle 95% of the "
        "resulting averages (percentile bootstrap).")
    out("p-value of the permutation test: the share of {n:,} random reassignments of "
        "the games to low and high stakes that give a difference at least as large "
        "as the one observed. A small value means the difference is unlikely to "
        "arise by chance alone.".format(n=N_RESAMPLES))
    out("")

    summary_rows = []
    for game_type in settings.GAME_TYPES:
        sub = games_df[games_df.game_type == game_type]
        if sub.empty:
            continue
        out(NAMES[game_type].upper())
        out("-" * 60)
        out(f"Measure: {METRIC[game_type]}")
        values = {}
        for stakes in ("low", "high"):
            s = sub[sub.stakes == stakes]
            if s.empty:
                continue
            lo, hi = bootstrap_ci(s.level, rng)
            values[stakes] = s.level.values
            row = {
                "game": NAMES[game_type], "stakes": stakes, "n_games": len(s),
                "mean_rounds": round(s.n_rounds.mean(), 2),
                "mean_level": round(s.level.mean(), 4),
                "sd_level": round(s.level.std(ddof=1), 4) if len(s) > 1 else np.nan,
                "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                "mean_first_round": round(s.first_round.mean(), 4),
                "mean_total_payoff_per_player": round(s.total_payoff_mean.mean(), 2),
            }
            if game_type == "pd":
                row["mean_mutual_cooperation"] = round(s.mutual_cooperation.mean(), 4)
            summary_rows.append(row)
            out(f"{stakes.capitalize():>5} stakes: {len(s)} games, mean length "
                f"{s.n_rounds.mean():.1f} rounds. Average {s.level.mean():.3f} "
                f"(95% confidence interval {lo:.3f} to {hi:.3f}); first round "
                f"{s.first_round.mean():.3f}.")
            if game_type == "pd":
                out(f"            Share of rounds in which both players chose the "
                    f"cooperative option: {s.mutual_cooperation.mean():.3f}.")
        if len(values) == 2:
            diff, (dlo, dhi), p = compare(values["low"], values["high"], rng)
            out(f"High minus low: {diff:+.3f} (95% confidence interval {dlo:+.3f} "
                f"to {dhi:+.3f}); permutation test p-value {p:.4f}.")
        # How a player reacts to the other player's previous round.
        rr = rounds_df[(rounds_df.game_type == game_type) & rounds_df.other_prev.notna()]
        if game_type == "pd":
            out("Reaction to the other player's previous choice (rounds 2 onwards, "
                "all players pooled; rounds within a game are not independent):")
            for stakes in ("low", "high"):
                s = rr[rr.stakes == stakes]
                if s.empty:
                    continue
                a = s[s.other_prev == 1.0].x
                b = s[s.other_prev == 0.0].x
                out(f"  {stakes.capitalize():>5} stakes: cooperation rate after the "
                    f"other player cooperated {a.mean():.3f} (n={len(a)}); after the "
                    f"other player did not {b.mean():.3f} (n={len(b)}).")
        else:
            out("Reaction to the other player's previous contribution (rounds 2 "
                "onwards): correlation between a player's contribution share and "
                "the other player's share in the previous round.")
            for stakes in ("low", "high"):
                s = rr[rr.stakes == stakes]
                if len(s) > 2 and s.x.std() > 0 and s.other_prev.std() > 0:
                    out(f"  {stakes.capitalize():>5} stakes: correlation "
                        f"{np.corrcoef(s.x, s.other_prev)[0, 1]:+.3f} (n={len(s)}).")
                else:
                    out(f"  {stakes.capitalize():>5} stakes: not enough variation.")
        out("")

    out("THEORY BENCHMARKS (for reading the numbers)")
    out("-" * 60)
    if "pd" in set(games_df.game_type):
        out("Prisoner's dilemma: in a single round, choosing the non-cooperative option "
            "is best whatever the other player does. In a repeated game whose end is "
            "left to chance alone, cooperation can be sustained, because a player can "
            "stop cooperating after a betrayal. In this study the game also always "
            f"ends after round {settings.MAX_ROUNDS}, and the players are told so. "
            "When the last possible round is known, the single-round argument applies "
            "to that round, then to the round before it, and so on, so strict theory "
            "predicts betrayal in every round. In practice only about 1.6% of games "
            "reach round 40, and reasoning backward over many rounds is not expected "
            "of players, so a high cooperation rate is not an error.")
    if "pgg" in set(games_df.game_type):
        out("Public goods game: each pound contributed costs the contributor 1 pound "
            f"and returns {settings.PGG_MULTIPLIER / settings.N_PLAYERS:g} pounds to the "
            "contributor, so contributing nothing is best in a single round. Both "
            "players contributing everything gives each of them more than both "
            "contributing nothing.")
    out("")

    if calls:
        pt = sum(c["prompt_tokens"] for c in calls)
        ct = sum(c["completion_tokens"] for c in calls)
        rt = sum(c["reasoning_tokens"] for c in calls)
        cost = (pt * settings.PRICE_INPUT_PER_M + ct * settings.PRICE_OUTPUT_PER_M) / 1e6
        out("USE OF THE MODEL")
        out("-" * 60)
        out(f"Calls: {len(calls):,}. Input tokens: {pt:,}. Output tokens: {ct:,} "
            f"(of which hidden reasoning: {rt:,}). Estimated cost: ${cost:.2f} "
            f"(prices in settings.py).")

    report = "\n".join(lines)
    (out_dir / "part_a_report.txt").write_text(report + "\n", encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(out_dir / "part_a_summary.csv", index=False)
    draw_chart(rounds_df, out_dir / "part_a_by_round.png")
    print(report)
    print(f"\nFiles written to {out_dir}")


if __name__ == "__main__":
    main()
