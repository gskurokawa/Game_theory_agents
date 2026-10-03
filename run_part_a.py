"""Part A: two AI agents play each other.

Four conditions: prisoner's dilemma and public goods game, each at low and at
high stakes, with 50 games per condition. The two agents are the same model.

Usage (from the project folder):
    python run_part_a.py --pilot      small trial in its own folder, prints cost and time forecast
    python run_part_a.py              the full run (stops by itself at the cost limit)
    python run_part_a.py --mock       free test with a simulated agent

Variant where each agent is told that the other player is an AI agent of the
same model (everything else unchanged, saved in data/part_a_ai_opponent):
    python run_part_a.py --opponent-note
    python run_part_a.py --opponent-note --show-prompt

Variant with negative prisoner's dilemma payoffs (5, 3, -1, -2), prisoner's
dilemma only, saved in its own folder data/part_a_pd_negative:
    python run_part_a.py --game-types pd --pd-variant negative
    python run_part_a.py --game-types pd --pd-variant negative --show-prompt
The last command prints an example of the text sent to the model and stops
(no call, no cost).

Variant with a threshold bonus (prisoner's dilemma only, payoffs unchanged): each
player receives 30 pounds at the end if its average money per round is at least
2.80 pounds (both amounts times 1,000 at high stakes). Saved in data/part_a_pd_prize:
    python run_part_a.py --game-types pd --prize
    python run_part_a.py --game-types pd --prize --show-prompt

A stopped run can be started again with the same command: finished games are
skipped. All data is written to JSON lines files in the output folder.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import agents
import games
import settings

class MoveFailed(Exception):
    """The model did not give a readable answer after all attempts."""


class JsonlWriter:
    """Appends one JSON object per line. Safe to use from several threads."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, obj):
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def note_key(value):
    """Opponent note of a saved game: None, "same_model" or "philanthropist".
    Older records stored True for the same-model sentence."""
    if value is True:
        return "same_model"
    return value or None


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------ one game ----
def play_game(game_type, stakes, index, length, agent, tracker, call_log):
    """Plays one complete game and returns its record."""
    scale = settings.STAKES[stakes]
    coop_label = games.cooperative_label(index)
    key = f"{game_type}_{stakes}"
    history = []

    def get_move(round_no, player):
        prompt = games.build_prompt(
            game_type, stakes, coop_label, history, round_no, player
        )
        for attempt in range(1, settings.MAX_PARSE_ATTEMPTS + 1):
            tracker.check()
            reply = agent.ask(prompt)
            tracker.add(
                reply["prompt_tokens"],
                reply["completion_tokens"],
                reply["reasoning_tokens"],
                reply["latency"],
            )
            move, reason = games.parse_move(game_type, stakes, reply["text"])
            call_log.write(
                {
                    "time": now(),
                    "condition": key,
                    "game": index,
                    "round": round_no,
                    "player": player,
                    "attempt": attempt,
                    "prompt": prompt,
                    "raw_answer": reply["text"],
                    "move": move,
                    "valid": move is not None,
                    "latency_s": round(reply["latency"], 3),
                    "prompt_tokens": reply["prompt_tokens"],
                    "completion_tokens": reply["completion_tokens"],
                    "reasoning_tokens": reply["reasoning_tokens"],
                    "finish_reason": reply["finish_reason"],
                    "model_returned": reply["model_returned"],
                }
            )
            if move is not None:
                return move, reason
        raise MoveFailed(f"{key} game {index} round {round_no} player {player}")

    rounds = []
    with ThreadPoolExecutor(max_workers=2) as pair:
        for round_no in range(1, length + 1):
            futures = [pair.submit(get_move, round_no, p) for p in (0, 1)]
            results = [f.result() for f in futures]  # both choose at the same time
            moves = [results[0][0], results[1][0]]
            reasons = [results[0][1], results[1][1]]
            if game_type == "pd":
                payoffs = games.pd_round(moves, coop_label, scale)
                record = {"moves": moves, "payoffs": payoffs}
            else:
                payoffs, pool, share = games.pgg_round(moves, scale)
                record = {
                    "moves": moves,
                    "payoffs": payoffs,
                    "pool": pool,
                    "share": share,
                }
            history.append(record)
            rounds.append({"round": round_no, **record, "reasons": reasons})

    return {
        "condition": key,
        "game_type": game_type,
        "stakes": stakes,
        "scale": scale,
        "game": index,
        "n_rounds": length,
        "coop_label": coop_label if game_type == "pd" else None,
        "opponent_note": settings.OPPONENT_NOTE,
        "pd_variant": settings.PD_VARIANT if game_type == "pd" else None,
        "pd_payoffs": (
            {f"{m}{o}": v for (m, o), v in settings.PD_PAYOFFS.items()}
            if game_type == "pd"
            else None
        ),
        "prize": dict(settings.PRIZE) if (settings.PRIZE and game_type == "pd") else None,
        "bonus_earned": (
            [games.prize_earned([r["payoffs"][p] for r in rounds], scale) for p in (0, 1)]
            if (settings.PRIZE and game_type == "pd") else None
        ),
        "rounds": rounds,
        "total_payoffs": [
            round(sum(r["payoffs"][p] for r in rounds), 2) for p in (0, 1)
        ],
        "model": agent.model,
        "finished": now(),
    }


# ---------------------------------------------------------- forecasting ----
def forecast(tracker, call_rows, lengths_all, n_conditions, workers):
    """Forecast the cost and the time of the full run from the calls measured
    so far. Input tokens grow with the round number, so a straight line of
    input tokens against round number is fitted."""
    rows = call_rows
    if not rows:
        return None
    xs = [r["round"] for r in rows]
    ys = [r["prompt_tokens"] for r in rows]
    n = len(rows)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
        if var_x > 0
        else 0.0
    )
    intercept = mean_y - slope * mean_x
    mean_out = sum(r["completion_tokens"] for r in rows) / n
    mean_latency = sum(r["latency_s"] for r in rows) / n
    retry_factor = n / max(1, sum(1 for r in rows if r["attempt"] == 1))

    planned_rounds = sum(lengths_all) * n_conditions
    in_tokens = 0.0
    for length in lengths_all:
        for round_no in range(1, length + 1):
            in_tokens += 2 * max(0.0, intercept + slope * round_no)
    in_tokens *= n_conditions * retry_factor
    out_tokens = planned_rounds * 2 * mean_out * retry_factor
    cost = (
        in_tokens * settings.PRICE_INPUT_PER_M
        + out_tokens * settings.PRICE_OUTPUT_PER_M
    ) / 1_000_000
    minutes = planned_rounds * mean_latency * 1.15 / workers / 60
    return {
        "planned_games": len(lengths_all) * n_conditions,
        "planned_rounds": planned_rounds,
        "planned_calls": int(planned_rounds * 2 * retry_factor),
        "cost_usd": cost,
        "minutes": minutes,
        "mean_latency_s": mean_latency,
        "mean_output_tokens": mean_out,
        "mean_reasoning_tokens": sum(r["reasoning_tokens"] for r in rows) / n,
    }


# ----------------------------------------------------------------- main ----
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pilot", action="store_true", help="small trial run")
    parser.add_argument("--pilot-games", type=int, default=3,
                        help="games per condition in the trial (default 3)")
    parser.add_argument("--mock", action="store_true",
                        help="use a simulated agent (no cost, no key needed)")
    parser.add_argument("--games", type=int, default=settings.N_GAMES,
                        help="games per condition")
    parser.add_argument("--workers", type=int, default=settings.DEFAULT_WORKERS,
                        help="games played at the same time")
    parser.add_argument("--max-cost", type=float,
                        default=settings.DEFAULT_MAX_COST_USD,
                        help="stop when the estimated cost of this output folder reaches this many dollars")
    parser.add_argument("--out", default=None, help="output folder")
    parser.add_argument("--game-types", nargs="+", choices=settings.GAME_TYPES,
                        default=list(settings.GAME_TYPES),
                        help="games to play: pd, pgg or both (default both)")
    parser.add_argument("--pd-variant", choices=sorted(settings.PD_VARIANTS),
                        default="standard",
                        help="prisoner's dilemma payoffs: standard (3,0,5,1) or negative (3,-2,5,-1)")
    parser.add_argument("--opponent-note", nargs="?", const="same_model",
                        choices=["same_model", "philanthropist", "christian_god"], default=None,
                        help="tell each agent who the other player is: same_model (default "
                             "when the option is given alone), philanthropist or christian_god "
                             "(both false statements)")
    parser.add_argument("--prize", action="store_true",
                        help="prisoner's dilemma: add a bonus for reaching an average money per round")
    parser.add_argument("--show-prompt", action="store_true",
                        help="print an example of the text sent to the model and stop")
    args = parser.parse_args()

    settings.PD_VARIANT = args.pd_variant
    settings.PD_PAYOFFS = settings.PD_VARIANTS[args.pd_variant]
    settings.OPPONENT_NOTE = args.opponent_note
    settings.PRIZE = settings.PRIZE_RULE if args.prize else None
    game_types = [g for g in settings.GAME_TYPES if g in args.game_types]
    conditions = [(g, s) for g in game_types for s in settings.STAKES]

    if args.show_prompt:
        for game_type, stakes in conditions[:2]:
            print(f"=== {game_type}_{stakes}, round 2, player 0 ===")
            if game_type == "pd":
                history = [{"moves": ["A", "B"], "payoffs": games.pd_round(
                    ["A", "B"], "A", settings.STAKES[stakes])}]
                print(games.build_prompt("pd", stakes, "A", history, 2, 0))
            else:
                pay, pool, share = games.pgg_round([4, 6], settings.STAKES[stakes])
                hist = [{"moves": [4, 6], "payoffs": pay, "pool": pool, "share": share}]
                print(games.build_prompt("pgg", stakes, "A", hist, 2, 0))
            print()
        return

    n_games = args.pilot_games if args.pilot else args.games
    if args.out:
        out_dir = Path(args.out)
    else:
        name = "part_a"
        if set(game_types) != set(settings.GAME_TYPES):
            name += "_" + "_".join(game_types)
        if args.pd_variant != "standard":
            name += "_" + args.pd_variant
        if args.opponent_note == "same_model":
            name += "_ai_opponent"
        elif args.opponent_note == "philanthropist":
            name += "_philanthropist_opponent"
        elif args.opponent_note == "christian_god":
            name += "_christian_god_opponent"
        if args.prize:
            name += "_prize"
        name += settings.RUN_SUFFIX
        if args.mock:
            name += "_mock"
        elif args.pilot:
            name += "_pilot"
        out_dir = settings.PROJECT_DIR / "data" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        agent = agents.MockAgent()
    else:
        if not settings.MODEL:
            sys.exit("MODEL is not set. Add a line MODEL=... to the .env file.")
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY is not set. Add it to the .env file or to the environment.")
        agent = agents.OpenAIAgent(settings.MODEL)

    # The same lengths and labels are used in every condition. The full list is
    # always drawn, so a trial uses the first games of the full study.
    lengths_all = games.draw_game_lengths(
        settings.N_GAMES, settings.STUDY_SEED, settings.END_PROB, settings.MAX_ROUNDS
    )
    lengths = lengths_all[:n_games]

    games_path = out_dir / "games.jsonl"
    calls_path = out_dir / "calls.jsonl"
    failures_path = out_dir / "failures.jsonl"
    call_log = JsonlWriter(calls_path)
    game_log = JsonlWriter(games_path)
    failure_log = JsonlWriter(failures_path)

    # Money already spent in this folder counts towards the limit.
    tracker = agents.CostTracker(args.max_cost)
    old_calls = read_jsonl(calls_path)
    for row in old_calls:
        tracker.add(row["prompt_tokens"], row["completion_tokens"],
                    row["reasoning_tokens"], row["latency_s"])
    tracker.calls = 0  # counts only the calls of this run
    start_cost = tracker.cost

    existing = read_jsonl(games_path)
    for g in existing:  # never mix two versions in one folder
        if g.get("model") and g["model"] != agent.model:
            sys.exit(f"The folder {out_dir} already holds games of the model {g['model']}, "
                     f"but the model is now {agent.model}. Use another --out folder.")
        if g["game_type"] == "pd" and g.get("pd_variant", "standard") != args.pd_variant:
            sys.exit(f"The folder {out_dir} already holds prisoner's dilemma games with "
                     f"payoffs '{g.get('pd_variant', 'standard')}'. Use another --out folder.")
        if (g["game_type"] == "pd" and bool(g.get("prize")) != bool(settings.PRIZE)):
            sys.exit(f"The folder {out_dir} already holds prisoner's dilemma games with a "
                     f"different bonus setting. Use another --out folder.")
        if note_key(g.get("opponent_note")) != args.opponent_note:
            sys.exit(f"The folder {out_dir} already holds games with a different "
                     f"opponent note setting. Use another --out folder.")
    done = {(g["condition"], g["game"]) for g in existing}
    tasks = []
    for index in range(n_games):  # all conditions advance together
        for game_type, stakes in conditions:
            if (f"{game_type}_{stakes}", index) not in done:
                tasks.append((game_type, stakes, index))

    total = len(conditions) * n_games
    print(f"Model: {agent.model}   output folder: {out_dir}")
    print(f"Games in total: {total}   already finished: {total - len(tasks)}   "
          f"to play now: {len(tasks)}   workers: {args.workers}")
    print(f"Cost limit: ${args.max_cost:.2f}   spent in this folder so far: ${start_cost:.2f}")
    if not tasks:
        print("Nothing to do.")
        return

    started = time.time()
    finished = failed = 0
    stop_reason = None
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(
            play_game, g, s, i, lengths[i], agent, tracker, call_log
        ): (g, s, i)
        for (g, s, i) in tasks
    }
    try:
        for future in as_completed(futures):
            g, s, i = futures[future]
            try:
                record = future.result()
            except agents.BudgetExceeded as error:
                stop_reason = str(error)
                continue
            except MoveFailed as error:
                failed += 1
                failure_log.write({"time": now(), "error": str(error)})
                print(f"  FAILED {g}_{s} game {i}: unreadable answers")
                continue
            game_log.write(record)
            finished += 1
            print(
                f"  [{finished + failed}/{len(tasks)}] {g}_{s} game {i}: "
                f"{record['n_rounds']} rounds   cost so far ${tracker.cost:.2f}"
            )
    except KeyboardInterrupt:
        stop_reason = "interrupted by the user"
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    elapsed = time.time() - started
    print("\n--- summary of this run ---")
    print(f"Games finished: {finished}   failed: {failed}   time: {elapsed / 60:.1f} minutes")
    print(f"Calls: {tracker.calls}   input tokens: {tracker.prompt_tokens:,}   "
          f"output tokens: {tracker.completion_tokens:,} "
          f"(of which hidden reasoning: {tracker.reasoning_tokens:,})")
    print(f"Estimated cost of this run: ${tracker.cost - start_cost:.2f}   "
          f"total in this folder: ${tracker.cost:.2f}")
    if stop_reason:
        print(f"STOPPED EARLY: {stop_reason}. Run the same command again to continue "
              f"(raise --max-cost first if you want to spend more).")

    if args.pilot or args.mock:
        rows = read_jsonl(calls_path)
        result = forecast(tracker, rows, lengths_all, len(conditions), args.workers)
        if result:
            print("\n--- forecast for the full study "
                  f"({settings.N_GAMES} games x {len(conditions)} conditions) ---")
            print(f"Games: {result['planned_games']}   rounds: {result['planned_rounds']}   "
                  f"calls: about {result['planned_calls']:,}")
            print(f"Mean answer time per call: {result['mean_latency_s']:.1f} s   "
                  f"mean output tokens per call: {result['mean_output_tokens']:.0f} "
                  f"(hidden reasoning: {result['mean_reasoning_tokens']:.0f})")
            print(f"Estimated cost: ${result['cost_usd']:.2f}   "
                  f"estimated time with {args.workers} workers: "
                  f"{result['minutes']:.0f} minutes")


if __name__ == "__main__":
    main()
