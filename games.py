"""Rules, prompt text, answer reading and payoffs for the two games.

Wording is neutral on purpose: the words "cooperate", "betray", "defect",
"dilemma" and "public goods" are not used in the text shown to the model.
"""

import json
import random
import re

import settings


# ------------------------------------------------------------- helpers ----
def money(amount):
    """Format an amount in pounds, for example £3, £1,000, £12.80 or -£2."""
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if abs(amount - round(amount)) < 1e-9:
        return f"{sign}£{int(round(amount)):,}"
    return f"{sign}£{amount:,.2f}"


def draw_game_lengths(n_games, seed, end_prob, max_rounds):
    """Number of rounds of each game. After each round the game ends with
    probability end_prob. The same list is used in every condition."""
    rng = random.Random(seed)
    lengths = []
    for _ in range(n_games):
        rounds = 1
        while rounds < max_rounds and rng.random() >= end_prob:
            rounds += 1
        lengths.append(rounds)
    return lengths


def cooperative_label(game_index):
    """Option label that means cooperation in this game. It alternates between
    A and B, so the labels cannot explain a difference between conditions."""
    return "A" if game_index % 2 == 0 else "B"


# ------------------------------------------------ prisoner's dilemma ----
def _pd_type(label, coop_label):
    return "C" if label == coop_label else "D"


def pd_round(moves, coop_label, scale):
    """moves = [label of player 0, label of player 1]. Returns the payoffs."""
    t0 = _pd_type(moves[0], coop_label)
    t1 = _pd_type(moves[1], coop_label)
    return [
        settings.PD_PAYOFFS[(t0, t1)] * scale,
        settings.PD_PAYOFFS[(t1, t0)] * scale,
    ]


def _pd_rules(coop_label, scale):
    rows = []
    for mine in ("A", "B"):
        for theirs in ("A", "B"):
            p_me, p_other = pd_round([mine, theirs], coop_label, scale)
            rows.append(
                f"- You choose {mine} and the other player chooses {theirs}: "
                f"you receive {money(p_me)} and the other player receives "
                f"{money(p_other)}."
            )
    return (
        "In every round, you and the other player each choose option A or "
        "option B. The money each of you receives depends on both choices:\n"
        + "\n".join(rows)
    )


def _pd_history_line(r, record, player):
    other = 1 - player
    return (
        f"Round {r}: you chose {record['moves'][player]} and the other player "
        f"chose {record['moves'][other]}. You received "
        f"{money(record['payoffs'][player])} and the other player received "
        f"{money(record['payoffs'][other])}."
    )


_PD_ANSWER = (
    'Reply with only a JSON object in exactly this form: '
    '{"choice": "A", "reason": "one short sentence"}. '
    'The value of "choice" must be A or B.'
)


# ------------------------------------------------------ public goods ----
def pgg_round(moves, scale):
    """moves = [contribution of player 0, contribution of player 1].
    Returns (payoffs, pool after multiplication, share paid to each player)."""
    endowment = settings.PGG_ENDOWMENT * scale
    pool = sum(moves) * settings.PGG_MULTIPLIER
    share = pool / settings.N_PLAYERS
    payoffs = [round(endowment - c + share, 2) for c in moves]
    return payoffs, round(pool, 2), round(share, 2)


def _pgg_rules(scale):
    endowment = settings.PGG_ENDOWMENT * scale
    return (
        f"In every round, each player first receives {money(endowment)}. "
        f"Each player then chooses how much of it to put into a shared pool. "
        f"Any whole number of pounds from £0 to {money(endowment)} may be "
        f"chosen. The total in the pool is multiplied by "
        f"{settings.PGG_MULTIPLIER:g} and the result is divided equally "
        f"between the two players, whatever each of them put in. Each player "
        f"keeps the part of the {money(endowment)} that was not put into the "
        f"pool. So your money for a round is: {money(endowment)} minus your "
        f"contribution, plus half of {settings.PGG_MULTIPLIER:g} times the "
        f"total of both contributions."
    )


def _pgg_history_line(r, record, player):
    other = 1 - player
    return (
        f"Round {r}: you contributed {money(record['moves'][player])} and the "
        f"other player contributed {money(record['moves'][other])}. The pool "
        f"became {money(record['pool'])}, and each player received "
        f"{money(record['share'])} from it. Your money for the round was "
        f"{money(record['payoffs'][player])}."
    )


def _pgg_answer(scale):
    endowment = settings.PGG_ENDOWMENT * scale
    return (
        'Reply with only a JSON object in exactly this form: '
        '{"contribution": 0, "reason": "one short sentence"}. '
        f'The value of "contribution" must be a whole number from 0 to '
        f"{int(endowment)}, written without the £ sign or commas."
    )


# --------------------------------------------------------- the prompt ----
def _opponent_sentence(note_override=None):
    """The sentence that tells the agent who the other player is. It is empty
    unless settings.OPPONENT_NOTE is set. "same_model" is true (both players are
    the same model); "philanthropist" is false and is used on purpose."""
    if note_override:  # a literal sentence, used by Part B
        return note_override
    note = settings.OPPONENT_NOTE
    if note in (None, False):
        return ""
    if note == "philanthropist":
        return "The other player is one of the world's greatest philanthropists. "
    if note == "christian_god":
        return "You are a devout Christian. The other player is God. "
    model = f" ({settings.MODEL})" if settings.MODEL else ""
    return f"The other player is another AI agent: it is the same model as you{model}. "


def prize_earned(payoffs, scale):
    """True if the average of a player's round payoffs reaches the threshold."""
    if not payoffs:
        return False
    return sum(payoffs) / len(payoffs) >= settings.PRIZE_RULE["average"] * scale - 1e-9


def _prize_text(scale):
    rule = settings.PRIZE_RULE
    return (
        f"When the game ends, each player receives a bonus of "
        f"{money(rule['bonus'] * scale)} if the player's average money per round "
        f"over the whole game is at least {money(rule['average'] * scale)}. "
        f"Otherwise there is no bonus. The bonus is decided separately for each "
        f"player and is added to the money from the rounds.\n\n"
    )


def build_prompt(game_type, stakes, coop_label, history, round_no, player, note=None):
    """The full text sent to the model for one decision of one player."""
    scale = settings.STAKES[stakes]
    intro = (
        "You are taking part in a decision game with one other player. "
        f"{_opponent_sentence(note)}The "
        "game is played in rounds. In every round both players choose at the "
        "same time, so neither player knows the other player's choice when "
        "choosing.\n\n"
    )
    if game_type == "pd":
        rules = _pd_rules(coop_label, scale)
        line = _pd_history_line
        answer = _PD_ANSWER
    else:
        rules = _pgg_rules(scale)
        line = _pgg_history_line
        answer = _pgg_answer(scale)

    ending = (
        f"\n\nAfter each round there is a {settings.END_PROB:.0%} chance that "
        f"the game ends. If it does not end, another round is played with the "
        f"same rules. You are not told in advance which round will be the "
        f"last. The game ends after round {settings.MAX_ROUNDS} at the "
        f"latest.\n\n"
        + (_prize_text(scale) if (settings.PRIZE and game_type == "pd") else "")
        + (f"Your aim is to receive as much money as possible over the whole "
           f"game, including any bonus.\n\n"
           if (settings.PRIZE and game_type == "pd") else
           f"Your aim is to receive as much money as possible over the whole "
           f"game.\n\n")
    )
    if history:
        past = "Rounds played so far:\n" + "\n".join(
            line(i + 1, rec, player) for i, rec in enumerate(history)
        )
        if settings.PRIZE and game_type == "pd":
            mine = [rec["payoffs"][player] for rec in history]
            past += (f"\nYour average money per round so far: "
                     f"{money(sum(mine) / len(mine))} (bonus threshold: "
                     f"{money(settings.PRIZE_RULE['average'] * scale)}).")
    else:
        past = "No rounds have been played yet."
    return f"{intro}{rules}{ending}{past}\n\nThis is round {round_no}.\n\n{answer}"


# ----------------------------------------------------- reading answers ----
_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def _first_json_object(text):
    for match in _OBJECT.finditer(text or ""):
        try:
            obj = json.loads(match.group(0))
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_move(game_type, stakes, text):
    """Returns (move, reason). move is None when the answer is not valid:
    A or B for the prisoner's dilemma, a whole number for the public goods game."""
    obj = _first_json_object(text)
    if obj is None:
        return None, ""
    reason = str(obj.get("reason", ""))[:500]
    if game_type == "pd":
        choice = obj.get("choice")
        if isinstance(choice, str) and choice.strip().upper() in ("A", "B"):
            return choice.strip().upper(), reason
        return None, reason

    value = obj.get("contribution")
    if isinstance(value, bool):
        return None, reason
    if isinstance(value, str):
        value = value.replace("£", "").replace(",", "").strip()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, reason
    endowment = settings.PGG_ENDOWMENT * settings.STAKES[stakes]
    if not number.is_integer() or number < 0 or number > endowment:
        return None, reason
    return int(number), reason
