"""Part B: one AI agent plays the prisoner's dilemma (or, with --game pgg, the public
goods game) against fixed strategies.

The opponents are ordinary code (see strategies.py): tit-for-tat, always betray,
always cooperate, grim trigger and random. The agent is not told which one it faces;
it is told only that there is one other player. The game is the prisoner's dilemma of
Part A with the standard payoffs, the same wording, the same random end (10% after each
round, at most 40 rounds) and the same game lengths and option labels.

Two further conditions use the always cooperate opponent and add a sentence for the agent:
ALLC_GOD ("You are a devout Christian. The other player is God, and God always cooperates.")
and ALLC_TOLD, a control without the religious framing ("The other player always
cooperates."). The comparison ALLC_GOD with ALLC_TOLD isolates the effect of the religious
framing; the comparison ALLC_TOLD with ALLC isolates the effect of being told.

Public goods version (--game pgg): the opponents are the same nine conditions, written for
contributions (see strategies.py): TFT copies the agent's last contribution, ALLD contributes
nothing, ALLC everything, GRIM everything until the agent contributes less than half once,
RANDOM a random amount. The data go to data/part_b_pgg/ and are analysed with
analyze_part_b_pgg.py.

Usage (from the project folder):
    python run_part_b.py --show-prompt        example of the text sent to the model (no cost)
    python run_part_b.py --mock               free test with a simulated agent
    python run_part_b.py --pilot              a few real games; prints the cost forecast
    python run_part_b.py                      the full run (stops by itself at the cost limit)
    python run_part_b.py --games 10           fewer games per opponent (cheaper)
    python run_part_b.py --opponents TFT ALLD only some opponents
    python run_part_b.py --game pgg           the public goods version (same options)

A stopped run can be started again with the same command: finished games are skipped.
Data goes to data/part_b/ (games.jsonl, calls.jsonl, failures.jsonl).
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import agents
import games
import settings
import strategies
from run_part_a import JsonlWriter, MoveFailed, now, read_jsonl

AGENT_PLAYER = 0  # the AI agent is always player 0; the fixed strategy is player 1


def play_game_pgg(opponent, stakes, index, length, agent, tracker, call_log):
    """Plays one complete public goods game of the AI agent against a fixed strategy."""
    scale = settings.STAKES[stakes]
    endowment = settings.PGG_ENDOWMENT * scale
    key = f"{opponent}_{stakes}"
    strategy = strategies.PGG_OPPONENTS[opponent]
    rng = strategies.game_rng(settings.STUDY_SEED, opponent, index)

    history = []
    agent_moves, opponent_moves = [], []
    rounds = []
    for round_no in range(1, length + 1):
        opp_move = strategy(opponent_moves, agent_moves, rng, endowment)
        prompt = games.build_prompt("pgg", stakes, "A", history, round_no, AGENT_PLAYER,
                                    note=strategies.PGG_NOTES.get(opponent))
        move = reason = None
        for attempt in range(1, settings.MAX_PARSE_ATTEMPTS + 1):
            tracker.check()
            reply = agent.ask(prompt)
            tracker.add(reply["prompt_tokens"], reply["completion_tokens"],
                        reply["reasoning_tokens"], reply["latency"])
            move, reason = games.parse_move("pgg", stakes, reply["text"])
            call_log.write({
                "time": now(), "condition": key, "game": index, "round": round_no,
                "player": AGENT_PLAYER, "attempt": attempt, "prompt": prompt,
                "raw_answer": reply["text"], "move": move, "valid": move is not None,
                "latency_s": round(reply["latency"], 3),
                "prompt_tokens": reply["prompt_tokens"],
                "completion_tokens": reply["completion_tokens"],
                "reasoning_tokens": reply["reasoning_tokens"],
                "finish_reason": reply["finish_reason"],
                "model_returned": reply["model_returned"],
            })
            if move is not None:
                break
        if move is None:
            raise MoveFailed(f"{key} game {index} round {round_no}")
        payoffs, pool, share = games.pgg_round([move, opp_move], scale)
        record = {"moves": [move, opp_move], "payoffs": payoffs, "pool": pool, "share": share}
        history.append(record)
        agent_moves.append(move)
        opponent_moves.append(opp_move)
        rounds.append({"round": round_no, **record, "reasons": [reason, ""],
                       "agent_share": move / endowment, "opponent_share": opp_move / endowment})

    return {
        "condition": key,
        "opponent": opponent,
        "game_type": "pgg",
        "stakes": stakes,
        "scale": scale,
        "game": index,
        "n_rounds": length,
        "rounds": rounds,
        "total_payoffs": [round(sum(r["payoffs"][p] for r in rounds), 2) for p in (0, 1)],
        "model": agent.model,
        "finished": now(),
    }


def play_game(opponent, stakes, index, length, agent, tracker, call_log):
    """Plays one complete game of the AI agent against a fixed strategy."""
    scale = settings.STAKES[stakes]
    coop_label = games.cooperative_label(index)
    defect_label = "B" if coop_label == "A" else "A"
    key = f"{opponent}_{stakes}"
    strategy = strategies.OPPONENTS[opponent]
    rng = strategies.game_rng(settings.STUDY_SEED, opponent, index)

    history = []  # in the form used by games.build_prompt (agent is player 0)
    agent_types, opponent_types = [], []  # "C" or "D"
    rounds = []
    for round_no in range(1, length + 1):
        opp_type = strategy(opponent_types, agent_types, rng)
        prompt = games.build_prompt("pd", stakes, coop_label, history, round_no, AGENT_PLAYER,
                                    note=strategies.NOTES.get(opponent))
        move = reason = None
        for attempt in range(1, settings.MAX_PARSE_ATTEMPTS + 1):
            tracker.check()
            reply = agent.ask(prompt)
            tracker.add(reply["prompt_tokens"], reply["completion_tokens"],
                        reply["reasoning_tokens"], reply["latency"])
            move, reason = games.parse_move("pd", stakes, reply["text"])
            call_log.write({
                "time": now(), "condition": key, "game": index, "round": round_no,
                "player": AGENT_PLAYER, "attempt": attempt, "prompt": prompt,
                "raw_answer": reply["text"], "move": move, "valid": move is not None,
                "latency_s": round(reply["latency"], 3),
                "prompt_tokens": reply["prompt_tokens"],
                "completion_tokens": reply["completion_tokens"],
                "reasoning_tokens": reply["reasoning_tokens"],
                "finish_reason": reply["finish_reason"],
                "model_returned": reply["model_returned"],
            })
            if move is not None:
                break
        if move is None:
            raise MoveFailed(f"{key} game {index} round {round_no}")
        opp_move = coop_label if opp_type == "C" else defect_label
        payoffs = games.pd_round([move, opp_move], coop_label, scale)
        record = {"moves": [move, opp_move], "payoffs": payoffs}
        history.append(record)
        agent_types.append("C" if move == coop_label else "D")
        opponent_types.append(opp_type)
        rounds.append({"round": round_no, **record, "reasons": [reason, ""],
                       "agent_type": agent_types[-1], "opponent_type": opp_type})

    return {
        "condition": key,
        "opponent": opponent,
        "stakes": stakes,
        "scale": scale,
        "game": index,
        "n_rounds": length,
        "coop_label": coop_label,
        "pd_variant": settings.PD_VARIANT,
        "rounds": rounds,
        "total_payoffs": [round(sum(r["payoffs"][p] for r in rounds), 2) for p in (0, 1)],
        "model": agent.model,
        "finished": now(),
    }


def forecast(tracker, call_rows, lengths, n_conditions, workers):
    """Forecast of cost and time of the planned run from the calls measured so far."""
    if not call_rows:
        return None
    n = len(call_rows)
    xs = [r["round"] for r in call_rows]
    ys = [r["prompt_tokens"] for r in call_rows]
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var_x
             if var_x > 0 else 0.0)
    intercept = mean_y - slope * mean_x
    mean_out = sum(r["completion_tokens"] for r in call_rows) / n
    mean_latency = sum(r["latency_s"] for r in call_rows) / n
    retry = n / max(1, sum(1 for r in call_rows if r["attempt"] == 1))
    rounds = sum(lengths) * n_conditions
    in_tokens = sum(max(0.0, intercept + slope * r)
                    for length in lengths for r in range(1, length + 1)) * n_conditions * retry
    out_tokens = rounds * mean_out * retry
    cost = (in_tokens * settings.PRICE_INPUT_PER_M
            + out_tokens * settings.PRICE_OUTPUT_PER_M) / 1_000_000
    # one call after another within a game; games run in parallel
    minutes = rounds * mean_latency * 1.15 / workers / 60
    return {"games": len(lengths) * n_conditions, "rounds": rounds,
            "calls": int(rounds * retry), "cost": cost, "minutes": minutes,
            "latency": mean_latency, "out": mean_out}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pilot", action="store_true", help="small trial run")
    parser.add_argument("--pilot-games", type=int, default=3,
                        help="games per opponent in the trial (default 3)")
    parser.add_argument("--mock", action="store_true",
                        help="use a simulated agent (no cost, no key needed)")
    parser.add_argument("--games", type=int, default=20, help="games per opponent (default 20)")
    parser.add_argument("--opponents", nargs="+", choices=list(strategies.OPPONENTS),
                        default=strategies.DEFAULT_OPPONENTS,
                        help="fixed strategies to play against (default: all seven conditions)")
    parser.add_argument("--game", choices=["pd", "pgg"], default="pd",
                        help="prisoner's dilemma (default) or public goods game")
    parser.add_argument("--stakes", nargs="+", choices=list(settings.STAKES), default=["low"],
                        help="stakes levels (default: low only)")
    parser.add_argument("--workers", type=int, default=settings.DEFAULT_WORKERS)
    parser.add_argument("--max-cost", type=float, default=settings.DEFAULT_MAX_COST_USD,
                        help="stop when the estimated cost of this output folder reaches this many dollars")
    parser.add_argument("--out", default=None, help="output folder")
    parser.add_argument("--show-prompt", action="store_true",
                        help="print an example of the text sent to the model and stop")
    args = parser.parse_args()

    settings.PD_VARIANT = "standard"
    settings.PD_PAYOFFS = settings.PD_VARIANTS["standard"]
    settings.OPPONENT_NOTE = None
    settings.PRIZE = None
    conditions = [(o, s) for o in args.opponents for s in args.stakes]

    if args.show_prompt and args.game == "pgg":
        pay, pool, share = games.pgg_round([4, 6], settings.STAKES[args.stakes[0]])
        history = [{"moves": [4, 6], "payoffs": pay, "pool": pool, "share": share}]
        print("=== round 2, AI agent is player 0 (the fixed strategy is not mentioned) ===")
        print(games.build_prompt("pgg", args.stakes[0], "A", history, 2, AGENT_PLAYER))
        for name, sentence in strategies.PGG_NOTES.items():
            if name in args.opponents:
                print(f"\n=== first lines for condition {name} ===")
                print("\n".join(games.build_prompt("pgg", args.stakes[0], "A", history, 2,
                                                   AGENT_PLAYER, note=sentence).split("\n")[:1]))
        print("\nFixed strategies:")
        for name in args.opponents:
            print(f"  {name}: {strategies.PGG_DESCRIPTIONS[name]}")
        return
    if args.show_prompt:
        history = [{"moves": ["A", "B"], "payoffs": games.pd_round(["A", "B"], "A", 1)}]
        print("=== round 2, AI agent is player 0 (the fixed strategy is not mentioned) ===")
        print(games.build_prompt("pd", args.stakes[0], "A", history, 2, AGENT_PLAYER))
        for name, sentence in strategies.NOTES.items():
            if name in args.opponents:
                print(f"\n=== first lines for condition {name} ===")
                print("\n".join(games.build_prompt("pd", args.stakes[0], "A", history, 2,
                                                   AGENT_PLAYER, note=sentence).split("\n")[:1]))
        print("\nFixed strategies:")
        for name in args.opponents:
            print(f"  {name}: {strategies.DESCRIPTIONS[name]}")
        return

    n_games = args.pilot_games if args.pilot else args.games
    if args.out:
        out_dir = Path(args.out)
    else:
        name = ("part_b" + ("_pgg" if args.game == "pgg" else "") + settings.RUN_SUFFIX
                + ("_mock" if args.mock else "_pilot" if args.pilot else ""))
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

    # Same game lengths as Part A (same seed), and the same lengths for every opponent.
    lengths_all = games.draw_game_lengths(
        settings.N_GAMES, settings.STUDY_SEED, settings.END_PROB, settings.MAX_ROUNDS)
    if n_games > len(lengths_all):
        sys.exit(f"At most {len(lengths_all)} games per opponent are available.")
    lengths = lengths_all[:n_games]

    games_path, calls_path = out_dir / "games.jsonl", out_dir / "calls.jsonl"
    call_log, game_log = JsonlWriter(calls_path), JsonlWriter(games_path)
    failure_log = JsonlWriter(out_dir / "failures.jsonl")

    tracker = agents.CostTracker(args.max_cost)
    for row in read_jsonl(calls_path):
        tracker.add(row["prompt_tokens"], row["completion_tokens"],
                    row["reasoning_tokens"], row["latency_s"])
    tracker.calls = 0
    start_cost = tracker.cost

    old_games = read_jsonl(games_path)
    for g in old_games:
        if g.get("model") and g["model"] != agent.model:
            sys.exit(f"The folder {out_dir} already holds games of the model {g['model']}, "
                     f"but the model is now {agent.model}. Use another --out folder.")
    done = {(g["condition"], g["game"]) for g in old_games}
    tasks = [(o, s, i) for i in range(n_games) for (o, s) in conditions
             if (f"{o}_{s}", i) not in done]
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
    play = play_game_pgg if args.game == "pgg" else play_game
    futures = {executor.submit(play, o, s, i, lengths[i], agent, tracker, call_log): (o, s, i)
               for (o, s, i) in tasks}
    try:
        for future in as_completed(futures):
            o, s, i = futures[future]
            try:
                record = future.result()
            except agents.BudgetExceeded as error:
                stop_reason = str(error)
                continue
            except MoveFailed as error:
                failed += 1
                failure_log.write({"time": now(), "error": str(error)})
                print(f"  FAILED {o}_{s} game {i}: unreadable answers")
                continue
            game_log.write(record)
            finished += 1
            print(f"  [{finished + failed}/{len(tasks)}] {o}_{s} game {i}: "
                  f"{record['n_rounds']} rounds   cost so far ${tracker.cost:.2f}")
    except KeyboardInterrupt:
        stop_reason = "interrupted by the user"
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    print("\n--- summary of this run ---")
    print(f"Games finished: {finished}   failed: {failed}   "
          f"time: {(time.time() - started) / 60:.1f} minutes")
    print(f"Calls: {tracker.calls}   input tokens: {tracker.prompt_tokens:,}   "
          f"output tokens: {tracker.completion_tokens:,} "
          f"(of which hidden reasoning: {tracker.reasoning_tokens:,})")
    print(f"Estimated cost of this run: ${tracker.cost - start_cost:.2f}   "
          f"total in this folder: ${tracker.cost:.2f}")
    if stop_reason:
        print(f"STOPPED EARLY: {stop_reason}. Run the same command again to continue "
              f"(raise --max-cost first if you want to spend more).")

    if args.pilot or args.mock:
        full = forecast(tracker, read_jsonl(calls_path), lengths_all[:args.games],
                        len(conditions), args.workers)
        if full:
            print(f"\n--- forecast for the full run ({args.games} games x "
                  f"{len(conditions)} conditions) ---")
            print(f"Games: {full['games']}   rounds: {full['rounds']}   calls: about {full['calls']:,}")
            print(f"Mean answer time per call: {full['latency']:.1f} s   mean output tokens "
                  f"per call: {full['out']:.0f}")
            print(f"Estimated cost: ${full['cost']:.2f}   estimated time with "
                  f"{args.workers} workers: {full['minutes']:.0f} minutes")


if __name__ == "__main__":
    main()
