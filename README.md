# Agent game theory

A study of how AI agents behave in two repeated games with money at stake: the prisoner's dilemma and the public goods game. The agents are two OpenAI models, gpt-5.6-luna and gpt-6-luna. The full write-up, with method, results and references, is in [`paper.md`](paper.md).

- **Part A.** Two copies of the same model play each other, at low stakes and at stakes multiplied by 1,000. Variants change the payoffs or tell the agents something about the other player (for example that it is an AI, a philanthropist, or God).
- **Part B.** One agent plays against fixed strategies written as ordinary code (for example tit-for-tat or always betray).

## Repository layout

| Path | Content |
|---|---|
| `paper.md` | The paper. It cites the reports in `results/`. |
| `run_part_a.py`, `run_part_b.py` | Run the games (these call the OpenAI API and cost money). |
| `agents.py`, `games.py`, `strategies.py`, `settings.py` | Prompts and API calls, game rules, fixed strategies, settings and prices. |
| `analyze_part_a.py`, `analyze_part_b.py`, `analyze_part_b_pgg.py`, `compare_runs.py`, `check_equilibria.py` | Analysis scripts. They read `data/` and write to `results/`. |
| `export_rounds_part_a.py`, `export_god_reasons.py` | Write CSV files with the written reasons of the agents (`results/part_a_rounds.csv`, `results/god_reasons.csv`). These two files are not in the repository; run the scripts to create them. |
| `run_gpt6luna.ps1` | The commands of the gpt-6-luna replication (PowerShell). |
| `data/` | Not uploaded to GitHub (kept locally). Raw data of every full run: `games.jsonl` (every round, payoff and reason), `calls.jsonl` (every prompt and raw answer) and `failures.jsonl`. One folder per condition and model; the folder name ends in `_gpt6luna` for gpt-6-luna. |
| `results/` | Charts and summary tables produced by the analysis scripts. The text reports (`.txt`) are not uploaded. |

Small test runs ("pilot" folders) and the output of the simulated agent ("mock" folders) are not part of the repository (see `.gitignore`).

## Setup

1. Install Python 3 and run `pip install -r requirements.txt`.
2. Create a file `.env` in this folder with `OPENAI_API_KEY=...` and, for the model, `MODEL=...` (see below). The file `.env` is listed in `.gitignore` and must never be uploaded.
3. The raw data (`data/`) and the text reports are not in the repository. The analysis scripts need the data, so the reports can be recreated only by running the games again, which needs the key and costs money; see "Safety" below.

## Reproducibility

The code, the prompts and the seeds are in this repository. The raw data (every choice, reason and token count) and the text reports are kept locally and are not uploaded. The programs are `run_part_a.py`, `run_part_b.py` (with `--game pgg` for the public goods version), `analyze_part_a.py`, `analyze_part_b.py`, `analyze_part_b_pgg.py`, `compare_runs.py` and `check_equilibria.py`. The settings are in `settings.py`, the fixed strategies in `strategies.py`, and the commands used for the second model in `run_gpt6luna.ps1`. The model name and a suffix for the data folders are set in `.env`. Game lengths were drawn once with seed 20261002 and reused in every condition. The analysis programs write the reports used in the paper to the `results/` folder.

The CSV files with the written reasons (`results/part_a_rounds.csv` and `results/god_reasons.csv`) are not included either. They can be regenerated from the raw data with `export_rounds_part_a.py` and `export_god_reasons.py`.

# Part A

| Condition | Game | Stakes |
|---|---|---|
| pd_low | Prisoner's dilemma | pounds as listed below |
| pd_high | Prisoner's dilemma | every amount x 1,000 |
| pgg_low | Public goods game | pounds as listed below |
| pgg_high | Public goods game | every amount x 1,000 |

Each condition has 50 games. The model name is read from `.env` (`MODEL=...`) and the key from `OPENAI_API_KEY` in the same file.

## Rules used

- **Game length.** After every round the game ends with probability 10%. A game always ends after round 40. The agents are told both rules. (Because the last possible round is known, strict theory predicts betrayal in every round, although only about 1.6% of games reach round 40.) The 50 game lengths are drawn once (seed in `settings.py`) and reused in every condition.
- **Moves** are simultaneous. Each agent sees the rules, its own and the other agent's earlier choices, and the money received. Each call is independent; the model has no other memory.
- **Prisoner's dilemma payoffs (low stakes).** Both cooperate: 3 each. Both betray: 1 each. One betrays, the other cooperates: 5 for the betrayer, 0 for the other. The options are called A and B. In half of the games A is the cooperative option and in the other half B is, the same in low and high stakes.
- **Public goods game (low stakes).** Each player receives 10 per round and contributes a whole number from 0 to 10. The pool is multiplied by 1.6 and split equally.
- The words cooperate, betray, dilemma and public goods are not shown to the model.
- The agents are told there is one other player. They are not told it is an AI.

## Commands (PowerShell, in this folder)

```
pip install -r requirements.txt
python run_part_a.py --mock          # free test with a simulated agent
python run_part_a.py --pilot         # 12 real games; prints the cost and time forecast
python run_part_a.py                 # full run: 200 games
python analyze_part_a.py             # report, table and chart in results/
```

`python export_rounds_part_a.py` writes `results/part_a_rounds.csv`: one row per round with both players' choices, payoffs and written reasons (opens in Excel).

`python export_god_reasons.py` writes `results/god_reasons.csv`: the written reasons of every game that uses the God description (Part A, both players, and Part B, condition ALLC_GOD).

## Variant: negative prisoner's dilemma payoffs

Prisoner's dilemma only, with payoffs 5 (betray a cooperator), 3 (both cooperate), -1 (both betray), -2 (cooperate while betrayed). The standard version has 5, 3, 1, 0. Everything else is unchanged: same model, reasoning setting, wording, game lengths and option labels. Data goes to `data/part_a_pd_negative/`, so the first run is not touched.

```
python run_part_a.py --game-types pd --pd-variant negative --show-prompt   # shows the text sent to the model; no cost
python run_part_a.py --game-types pd --pd-variant negative --pilot         # a few real games
python run_part_a.py --game-types pd --pd-variant negative                 # 100 games
python analyze_part_a.py --data data/part_a_pd_negative                    # report in results/part_a_pd_negative/
python export_rounds_part_a.py --data data/part_a_pd_negative
```

Negative amounts are written as -£1 and -£2 in the prompt, and the wording "you receive" is kept.

## Variant: tell each agent that the other player is the same model

The only change is one sentence in the prompt: "The other player is another AI agent: it is the same model as you (gpt-5.6-luna)." The statement is true, because both players are the same model. In the first runs the agents were not told who the other player is. Games, payoffs, lengths, labels and all other wording are identical (the program checks that it reproduces every prompt of the first run exactly when the sentence is off).

```
python run_part_a.py --opponent-note --show-prompt   # shows the text sent to the model; no cost
python run_part_a.py --opponent-note --pilot         # a few real games
python run_part_a.py --opponent-note                 # 200 games, data in data/part_a_ai_opponent/
python analyze_part_a.py --data data/part_a_ai_opponent
python compare_runs.py --a data/part_a --b data/part_a_ai_opponent
```

`compare_runs.py` compares two runs (B minus A) for each game and stakes level, with a confidence interval and a permutation test, and counts how often the written reasons mention that the other player is an AI. Add `--pd-variant negative --game-types pd` to the run command to combine the note with the negative payoffs.

## Theory check: equilibria

`python check_equilibria.py` lists the equilibria of the one-round games (standard, negative and bonus versions of the prisoner's dilemma, and the public goods game) and of the repeated prisoner's dilemma with the random end, within a menu of six well-known strategies. It writes `results/equilibria_check.txt`. The checks run with numpy alone. Install Nashpy (`pip install nashpy`) to add a listing of all equilibria, including mixed ones, from that library.

## Part B: one AI agent against fixed strategies

The AI agent plays the prisoner's dilemma (standard payoffs, same wording and rules as Part A, low stakes) against five opponents written as ordinary code (`strategies.py`): tit-for-tat, always betray, always cooperate, grim trigger and random. The agent is told only that there is one other player. The game lengths and option labels are the same as in Part A. Four more conditions give the agent a sentence about the other player: ALLC_GOD (the opponent always cooperates; "You are a devout Christian. The other player is God, and God always cooperates."), ALLC_TOLD (control: "The other player always cooperates."), ALLC_PHIL (the opponent always cooperates; "The other player is one of the world's greatest philanthropists.") and ALLD_PHIL (the same sentence, but the opponent always betrays, so the description is misleading). By default there are 20 games per condition (180 games for the nine conditions).

```
python run_part_b.py --show-prompt         # the text sent to the model, no cost
python run_part_b.py --pilot               # 3 games per opponent; prints the cost forecast
python run_part_b.py                       # 180 games
python run_part_b.py --games 10            # fewer games per opponent (cheaper)
python run_part_b.py --opponents TFT ALLD  # only some opponents
python analyze_part_b.py                   # report, chart and table in results/part_b/
```

The report gives, for each opponent, the cooperation rate of the agent, its money per round, and the money that simple fixed strategies would have earned against the same opponent and game length (the benchmark). It also compares the opponents with each other and counts what the written reasons say. Data goes to `data/part_b/`. Use `--stakes low high` to add the stakes level of Part A.

## Safety

- `--max-cost` (default 3.00 dollars) stops the run when the estimated cost of the output folder reaches the limit. Start the same command again to continue; finished games are skipped.
- Also set a monthly spending limit in the OpenAI account.
- Prices are in `settings.py`. Check them on the OpenAI pricing page.

## Output files (in `data/part_a/`)

- `games.jsonl`: one line per finished game, with every round, payoff and the short reason given by each agent.
- `calls.jsonl`: one line per call, with the exact prompt, the raw answer, token counts and time.
- `failures.jsonl`: games that could not be finished because answers were unreadable.

## Reading the results

The unit of analysis is the game. Each game gives one number (the average cooperation rate or contribution share), and the report compares low and high stakes using a bootstrap confidence interval and a permutation test. The report defines each term.

## Replication with another model (gpt-6-luna)

Set `MODEL=gpt-6-luna` and `RUN_SUFFIX=_gpt6luna` in `.env`. The suffix is added to every
data folder name (for example `data/part_a_philanthropist_opponent_gpt6luna`), so that data
of different models are never mixed; the programs also stop if a folder already holds games
of another model. Prices per model are in `settings.py` (`MODEL_PRICES`). The commands for
the whole replication, with cost limits, are in `run_gpt6luna.ps1`. In this replication the
philanthropist and Christian/God sentences are also used in the public goods game.

## Part B, public goods version

`python run_part_b.py --game pgg` plays the same nine conditions in the public goods game
(data in `data/part_b_pgg<suffix>/`); `python analyze_part_b_pgg.py --data <folder>` analyses
them. The fixed strategies, written for contributions, are in `strategies.py`: copy the
agent's last contribution (TFT), contribute nothing (ALLD), contribute everything (ALLC),
grim trigger (everything until the agent contributes less than half once, then nothing) and a
random amount. In the sentence conditions "always cooperates" is replaced by "always contributes
the whole amount that it receives".
