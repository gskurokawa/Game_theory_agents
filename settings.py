"""Settings for the study.

Every part of the program reads its values from this file, so that the same
rules, the same model and the same prices are used everywhere.
The model name and the OpenAI key are read from the file .env in this folder.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:  # python-dotenv is listed in requirements.txt
    pass

PROJECT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------- model ----
MODEL = os.getenv("MODEL", "").strip()  # for example gpt-5.6-luna (set in .env)
# Sampling temperature is NOT sent: some newer models only accept their default.
# Set to "minimal", "low", "medium" or "high" to limit hidden reasoning, or
# leave as None to use the model's default. Hidden reasoning is billed as output.
REASONING_EFFORT = "medium"
# Upper limit on the tokens the model may produce per answer. It includes the
# hidden reasoning tokens, so keep it generous.
MAX_OUTPUT_TOKENS = 3000

# Text added to the name of every data folder (for example _gpt6luna), so that runs
# with another model never mix with earlier data. Set RUN_SUFFIX in .env.
RUN_SUFFIX = os.getenv("RUN_SUFFIX", "").strip()

# --------------------------------------------------------------- prices ----
# US dollars per one million tokens (input, output), by model name. Published lists
# disagreed for gpt-5.6-luna (0.20/1.20 against 0.10/0.50); the higher pair was used
# for it. For gpt-6-luna the price is 0.10 input and 0.50 output (VentureBeat and
# the OpenAI model page, October 2026). Check the OpenAI pricing page.
MODEL_PRICES = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-6-luna": (0.10, 0.50),
}
PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M = MODEL_PRICES.get(MODEL, (0.20, 1.20))
# The run stops by itself when its estimated cost reaches this amount.
DEFAULT_MAX_COST_USD = 3.00

# ---------------------------------------------------------- study design ----
STUDY_SEED = 20261002  # fixes the game lengths, so a run can be repeated
N_GAMES = 50  # games per condition
END_PROB = 0.10  # chance that the game ends after each round
MAX_ROUNDS = 40  # a game always ends after this round (the agents are told)
STAKES = {"low": 1, "high": 1000}  # every amount is multiplied by this number
GAME_TYPES = ("pd", "pgg")  # prisoner's dilemma, public goods game
N_PLAYERS = 2

# Prisoner's dilemma payoffs in pounds at low stakes.
# Key: (my move, other player's move); C = cooperate, D = betray.
# "standard" is the version used in the first run of Part A.
# "negative" changes only two payoffs: mutual betrayal -1 (was 1) and being
# betrayed -2 (was 0). Both versions satisfy 5 > 3 > P > S, so betraying
# still pays more in a single round whatever the other player does.
PD_VARIANTS = {
    "standard": {("C", "C"): 3, ("C", "D"): 0, ("D", "C"): 5, ("D", "D"): 1},
    "negative": {("C", "C"): 3, ("C", "D"): -2, ("D", "C"): 5, ("D", "D"): -1},
}
PD_VARIANT = "standard"  # run_part_a.py sets this from --pd-variant
PD_PAYOFFS = PD_VARIANTS[PD_VARIANT]

# Sentence telling each agent who the other player is. None in the first runs of
# Part A. run_part_a.py sets this from --opponent-note.
#   "same_model": the other player is an AI agent of the same model (true).
#   "philanthropist": the other player is one of the world's greatest
#                     philanthropists (false; used to test the effect of a
#                     description of the other player).
#   "christian_god":  each agent is told that it is a devout Christian and that
#                     the other player is God (false; both agents get the same text).
OPPONENT_NOTE = None

# Threshold bonus for the prisoner's dilemma (None = no bonus, as in the first
# runs). When set, a player receives "bonus" pounds at the end of the game if the
# player's average money per round over the whole game is at least "average"
# pounds. Both amounts are multiplied by the stakes factor. run_part_a.py sets
# this from --prize.
PRIZE_RULE = {"bonus": 30, "average": 2.8}
PRIZE = None

# Public goods game at low stakes: each player receives this amount per round.
PGG_ENDOWMENT = 10
# The pool is multiplied by this number and split equally between the players.
PGG_MULTIPLIER = 1.6

# ------------------------------------------------------- reliability ----
MAX_PARSE_ATTEMPTS = 3  # asks again if the answer cannot be read
MAX_API_RETRIES = 6  # retries after a rate limit or a connection error
REQUEST_TIMEOUT_S = 180
DEFAULT_WORKERS = 8  # number of games played at the same time
