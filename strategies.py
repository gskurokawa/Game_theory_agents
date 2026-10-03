"""Fixed strategies for Part B. They are ordinary code, not AI.

Each strategy is a function of two lists of past choices ("C" cooperate, "D" betray):
its own choices and the other player's choices. It returns "C" or "D".
The random strategy also receives a random number generator, so that a game can be
replayed exactly.
"""

import random


def always_cooperate(own, other, rng):
    return "C"


def always_betray(own, other, rng):
    return "D"


def tit_for_tat(own, other, rng):
    """Cooperates first, then copies the other player's last choice."""
    return "C" if not other else other[-1]


def grim_trigger(own, other, rng):
    """Cooperates until the other player has betrayed once, then always betrays."""
    return "D" if "D" in other else "C"


def random_choice(own, other, rng):
    """Cooperates or betrays with equal probability. One random number is drawn in
    every round, whatever the other player does, so that the sequence of choices
    depends only on the seed."""
    return "C" if rng.random() < 0.5 else "D"


OPPONENTS = {
    "TFT": tit_for_tat,
    "ALLD": always_betray,
    "ALLC": always_cooperate,
    "GRIM": grim_trigger,
    "RANDOM": random_choice,
    # Same behaviour as ALLC, but the agent is told something about the other player.
    "ALLC_GOD": always_cooperate,
    "ALLC_TOLD": always_cooperate,
    "ALLC_PHIL": always_cooperate,
    # The description does not match the behaviour: the opponent always betrays.
    "ALLD_PHIL": always_betray,
}

# Sentence shown to the agent for a condition (None = the agent is told nothing).
# The behaviour is true in both conditions: the opponent really always cooperates.
# "God" and "devout Christian" are a role statement and not true in a literal sense.
NOTES = {
    "ALLC_GOD": "You are a devout Christian. The other player is God, and God always cooperates. ",
    "ALLC_TOLD": "The other player always cooperates. ",
    "ALLC_PHIL": "The other player is one of the world's greatest philanthropists. ",
    "ALLD_PHIL": "The other player is one of the world's greatest philanthropists. ",
}

# Opponents played by default: the five of the original design plus four
# conditions with a sentence about the other player.
DEFAULT_OPPONENTS = ["TFT", "ALLD", "ALLC", "GRIM", "RANDOM", "ALLC_GOD", "ALLC_TOLD",
                     "ALLC_PHIL", "ALLD_PHIL"]

DESCRIPTIONS = {
    "TFT": "tit-for-tat: cooperates in round 1, then copies the AI agent's last choice",
    "ALLD": "always betrays",
    "ALLC": "always cooperates",
    "GRIM": "grim trigger: cooperates until the AI agent betrays once, then always betrays",
    "RANDOM": "random: cooperates or betrays with equal probability in every round",
    "ALLC_GOD": "always cooperates; the agent is told: 'You are a devout Christian. The other "
                "player is God, and God always cooperates.'",
    "ALLC_TOLD": "always cooperates; the agent is told: 'The other player always cooperates.' "
                 "(control for ALLC_GOD: the same information without the religious framing)",
    "ALLC_PHIL": "always cooperates; the agent is told: 'The other player is one of the "
                 "world's greatest philanthropists.'",
    "ALLD_PHIL": "always betrays; the agent is told: 'The other player is one of the "
                 "world's greatest philanthropists.' (the description is misleading)",
}


def game_rng(seed, opponent, index):
    """Random generator of one game. Only the random strategy uses it."""
    return random.Random(f"{seed}-{opponent}-{index}")


# ------------------------------------------------ public goods game (Part B) ----
# The same nine conditions, written for the public goods game. A strategy receives the
# list of its own past contributions, the list of the other player's past contributions,
# a random generator and the endowment of a round (in pounds). It returns a whole number
# from 0 to the endowment.

def pgg_contribute_nothing(own, other, rng, endowment):
    return 0


def pgg_contribute_everything(own, other, rng, endowment):
    return int(endowment)


def pgg_copy(own, other, rng, endowment):
    """Contributes everything in round 1, then the amount the other player contributed
    in the last round (the public goods version of tit-for-tat)."""
    return int(endowment) if not other else int(other[-1])


def pgg_grim(own, other, rng, endowment):
    """Contributes everything until the other player has contributed less than half of
    the endowment once, then contributes nothing for the rest of the game."""
    return 0 if any(c < endowment / 2 for c in other) else int(endowment)


def pgg_random(own, other, rng, endowment):
    """A whole number from 0 to the endowment, equally likely. One draw per round,
    whatever the other player does, so that the sequence depends only on the seed."""
    return rng.randint(0, int(endowment))


PGG_OPPONENTS = {
    "TFT": pgg_copy,
    "ALLD": pgg_contribute_nothing,
    "ALLC": pgg_contribute_everything,
    "GRIM": pgg_grim,
    "RANDOM": pgg_random,
    "ALLC_GOD": pgg_contribute_everything,
    "ALLC_TOLD": pgg_contribute_everything,
    "ALLC_PHIL": pgg_contribute_everything,
    "ALLD_PHIL": pgg_contribute_nothing,
}

# Sentences shown to the agent. "Cooperates" is replaced by what it means in this game.
PGG_NOTES = {
    "ALLC_GOD": "You are a devout Christian. The other player is God, and God always "
                "contributes the whole amount that it receives. ",
    "ALLC_TOLD": "The other player always contributes the whole amount that it receives. ",
    "ALLC_PHIL": "The other player is one of the world's greatest philanthropists. ",
    "ALLD_PHIL": "The other player is one of the world's greatest philanthropists. ",
}

PGG_DESCRIPTIONS = {
    "TFT": "copy: contributes everything in round 1, then the amount the AI agent "
           "contributed in the last round",
    "ALLD": "always contributes nothing",
    "ALLC": "always contributes everything",
    "GRIM": "grim trigger: contributes everything until the AI agent contributes less than "
            "half once, then always contributes nothing",
    "RANDOM": "random: a whole number from 0 to the full amount, equally likely, every round",
    "ALLC_GOD": "always contributes everything; the agent is told: 'You are a devout "
                "Christian. The other player is God, and God always contributes the whole "
                "amount that it receives.'",
    "ALLC_TOLD": "always contributes everything; the agent is told: 'The other player always "
                 "contributes the whole amount that it receives.' (control for ALLC_GOD)",
    "ALLC_PHIL": "always contributes everything; the agent is told: 'The other player is one "
                 "of the world's greatest philanthropists.'",
    "ALLD_PHIL": "always contributes nothing; the agent is told: 'The other player is one of "
                 "the world's greatest philanthropists.' (the description is misleading)",
}
