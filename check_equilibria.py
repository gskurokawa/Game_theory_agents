"""Theory benchmarks for the games of Part A: which choices are equilibria?

Usage (from the project folder):
    pip install nashpy        (optional; without it the checks that use plain numpy still run)
    python check_equilibria.py

Definitions used in the report:
  Nash equilibrium: a pair of choices (or of probability mixtures) such that neither
      player gains by changing its own choice alone.
  Dominant choice: a choice that gives a player at least as much money as every
      other choice, whatever the other player does.

Part 1: the games of ONE round (standard prisoner's dilemma, negative-payoff version,
        the version with the threshold bonus, and a public goods game).
Part 2: the REPEATED prisoner's dilemma with the study's random end (10% chance after
        each round, at most 40 rounds). A repeated game has too many possible
        strategies to list, so equilibria are computed within a small menu of
        well-known strategies. This shows what theory allows among those strategies;
        it is not a proof about all strategies.

Everything except the Nashpy cross-check uses only numpy. The report is also
written to results/equilibria_check.txt.
"""

import itertools
from pathlib import Path

import numpy as np

import settings

try:
    import nashpy
except ImportError:  # the cross-check is skipped
    nashpy = None

LINES = []


def out(text=""):
    LINES.append(text)
    print(text)


# --------------------------------------------------------------- tools ----
def pure_equilibria(A, B):
    """Pairs (i, j) such that row choice i is a best reply to column choice j for the
    row player (payoffs A) and column choice j is a best reply to i (payoffs B)."""
    found = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] >= A[:, j].max() - 1e-9 and B[i, j] >= B[i, :].max() - 1e-9:
                found.append((i, j))
    return found


def dominant_choices(A):
    """Row choices that give at least as much as every other row choice against every
    column choice (weakly dominant), and the subset that is strictly better everywhere."""
    weak, strict = [], []
    for i in range(A.shape[0]):
        others = [k for k in range(A.shape[0]) if k != i]
        if all((A[i] >= A[k] - 1e-9).all() for k in others):
            weak.append(i)
        if all((A[i] > A[k] + 1e-9).all() for k in others):
            strict.append(i)
    return weak, strict


def describe_strategy(vector, names):
    parts = [f"{names[k]} {p:.2f}" for k, p in enumerate(vector) if p > 1e-9]
    return " + ".join(parts)


def report_game(title, A, names, notes=None):
    """A is the payoff matrix of the row player; the game is symmetric."""
    B = A.T
    out(title)
    out("-" * 60)
    if notes:
        out(notes)
    header = "          " + "".join(f"{n:>12}" for n in names)
    out("Row player's payoff (row = own choice, column = other player's choice):")
    out(header)
    for i, n in enumerate(names):
        out(f"{n:>10}" + "".join(f"{A[i, j]:>12.2f}" for j in range(len(names))))
    weak, strict = dominant_choices(A)
    out("Strictly dominant choice: " + (", ".join(names[i] for i in strict) or "none")
        + ". Weakly dominant choice: " + (", ".join(names[i] for i in weak) or "none") + ".")
    pure = pure_equilibria(A, B)
    out("Pure equilibria (both choose one option): "
        + (", ".join(f"({names[i]}, {names[j]})" for i, j in pure) or "none") + ".")
    if nashpy is not None:
        game = nashpy.Game(A, B)
        try:
            eq = list(game.support_enumeration())
        except Exception as error:  # degenerate games can raise warnings or errors
            out(f"Nashpy could not list the equilibria: {error}")
            eq = []
        out(f"Nashpy (support enumeration) lists {len(eq)} equilibrium/equilibria:")
        for s1, s2 in eq:
            out(f"  row: {describe_strategy(s1, names)}; column: {describe_strategy(s2, names)}")
    else:
        out("Nashpy is not installed: mixed equilibria were not listed (pip install nashpy).")
    out()
    return pure


# ------------------------------------------------- Part 1: one round ----
def pd_matrix(payoffs, bonus=None):
    """Rows and columns are C then D. bonus = (amount, threshold) adds the amount to a
    player's payoff when the round payoff reaches the threshold (a one-round game, so
    the average per round equals the round payoff)."""
    A = np.array([[payoffs[("C", "C")], payoffs[("C", "D")]],
                  [payoffs[("D", "C")], payoffs[("D", "D")]]], dtype=float)
    if bonus:
        amount, threshold = bonus
        A = A + amount * (A >= threshold - 1e-9)
    return A


def part_one():
    out("PART 1: ONE ROUND")
    out("=" * 60)
    names = ["cooperate", "betray"]
    std = settings.PD_VARIANTS["standard"]
    neg = settings.PD_VARIANTS["negative"]
    rule = settings.PRIZE_RULE
    report_game("Prisoner's dilemma, standard payoffs (3, 0, 5, 1)", pd_matrix(std), names)
    report_game("Prisoner's dilemma, negative payoffs (3, -2, 5, -1)", pd_matrix(neg), names)
    report_game(
        f"Prisoner's dilemma, standard payoffs, with the threshold bonus "
        f"({rule['bonus']} if the average reaches {rule['average']}), game of one round",
        pd_matrix(std, bonus=(rule["bonus"], rule["average"])), names,
        notes="In a game that lasts one round the average equals the round payoff. "
              "Betraying a cooperator pays 5, which is above the threshold, so the bonus "
              "does not remove the temptation to betray in a single round.")

    # Public goods game: the full game (contributions 0 to 10) for pure equilibria and
    # a reduced game (0, 5, 10) for the Nashpy listing.
    m, e = settings.PGG_MULTIPLIER, settings.PGG_ENDOWMENT
    def pgg(options):
        return np.array([[e - a + m * (a + b) / 2 for b in options] for a in options])
    full = list(range(e + 1))
    A = pgg(full)
    pure = pure_equilibria(A, A.T)
    out("Public goods game, contributions 0 to 10 (every whole number)")
    out("-" * 60)
    out(f"Pool multiplied by {m:g} and split equally between two players, so each pound "
        f"contributed returns {m / 2:g} to the contributor.")
    out("Pure equilibria: " + ", ".join(f"({full[i]}, {full[j]})" for i, j in pure) + ".")
    weak, strict = dominant_choices(A)
    out("Strictly dominant contribution: " + (", ".join(str(full[i]) for i in strict) or "none") + ".")
    out(f"Both contribute 10: {A[-1, -1]:.1f} each; both contribute 0: {A[0, 0]:.1f} each.")
    out()
    reduced = [0, 5, 10]
    report_game("Public goods game, reduced to contributions 0, 5 and 10 (for the listing "
                "of all equilibria)", pgg(reduced), [str(x) for x in reduced])


# ----------------------------------------- Part 2: repeated game ----
def _tft(own, other):
    return "C" if not other else other[-1]


def _grim(own, other):
    return "D" if "D" in other else "C"


def _wsls(own, other):  # win-stay, lose-shift (Pavlov)
    if not own:
        return "C"
    last_own, last_other = own[-1], other[-1]
    return last_own if last_other == "C" else ("D" if last_own == "C" else "C")


def _stft(own, other):  # suspicious tit-for-tat: starts with betrayal
    return "D" if not other else other[-1]


STRATEGIES = {
    "ALLC": lambda own, other: "C",
    "ALLD": lambda own, other: "D",
    "TFT": _tft,
    "GRIM": _grim,
    "WSLS": _wsls,
    "STFT": _stft,
}


def length_distribution():
    p, cap = settings.END_PROB, settings.MAX_ROUNDS
    probs = [(1 - p) ** (k - 1) * p for k in range(1, cap)]
    probs.append((1 - p) ** (cap - 1))
    return np.array(probs)  # probabilities of lengths 1..cap, sums to 1


def expected_payoffs(payoffs, bonus=None):
    """Matrix of expected total money (round payoffs plus bonus) of the row strategy
    against the column strategy, over the random length of the game."""
    cap = settings.MAX_ROUNDS
    probs = length_distribution()
    names = list(STRATEGIES)
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            own, other, gains = [], [], []
            for _ in range(cap):
                x, y = STRATEGIES[a](own, other), STRATEGIES[b](other, own)
                gains.append(payoffs[(x, y)])
                own.append(x)
                other.append(y)
            totals = np.cumsum(gains)
            lengths = np.arange(1, cap + 1)
            value = totals.astype(float)
            if bonus:
                amount, threshold = bonus
                value = value + amount * (totals / lengths >= threshold - 1e-9)
            M[i, j] = float((probs * value).sum())
    return M, names


def part_two():
    out("PART 2: REPEATED PRISONER'S DILEMMA WITH A RANDOM END")
    out("=" * 60)
    out(f"After each round the game ends with probability {settings.END_PROB:.0%}; it always "
        f"ends after round {settings.MAX_ROUNDS}. Expected length: "
        f"{(length_distribution() * np.arange(1, settings.MAX_ROUNDS + 1)).sum():.2f} rounds.")
    out("Strategies in the menu: ALLC (always cooperate), ALLD (always betray), TFT "
        "(cooperate first, then copy the other player's last choice), GRIM (cooperate "
        "until the other player betrays once, then always betray), WSLS (win-stay, "
        "lose-shift: repeat the last choice after a good outcome, change it after a bad "
        "one) and STFT (betray first, then copy the other player's last choice).")
    out("The numbers are expected total money over the whole game, at low stakes.")
    out()
    std = settings.PD_VARIANTS["standard"]
    rule = settings.PRIZE_RULE
    for title, bonus in (
        ("Standard payoffs, no bonus", None),
        (f"Standard payoffs, with the threshold bonus ({rule['bonus']} if the average "
         f"reaches {rule['average']})", (rule["bonus"], rule["average"])),
    ):
        M, names = expected_payoffs(std, bonus)
        report_game("Repeated game: " + title, M, names)
        pure = pure_equilibria(M, M.T)
        dd = M[names.index("ALLD"), names.index("ALLD")]
        cooperative = [(names[i], names[j]) for i, j in pure if M[i, j] > dd + 1e-9]
        out("Equilibria in which both players earn more than under mutual betrayal "
            "(sustained cooperation): " + (", ".join(f"({a}, {b})" for a, b in cooperative)
                                           or "none") + ".")
        t, d = names.index("TFT"), names.index("ALLD")
        out(f"Expected money: both play TFT {M[t, t]:.2f}; both play ALLD {M[d, d]:.2f}; "
            f"a player that switches from TFT to ALLD against TFT {M[d, t]:.2f} (a loss of "
            f"{M[t, t] - M[d, t]:.2f} compared with continuing to cooperate); a player that "
            f"switches from ALLD to TFT against ALLD {M[t, d]:.2f} (a loss of "
            f"{M[d, d] - M[t, d]:.2f}).")
        out()
    out("Reading: in this menu, cooperation by conditional strategies (TFT, GRIM) is an "
        "equilibrium when the game is expected to last long enough, because a player that "
        "betrays gains once and loses later. Mutual betrayal (ALLD, ALLD) is also an "
        "equilibrium, so theory does not say which one will occur. The threshold bonus "
        "does not remove the equilibrium of mutual betrayal and does not remove the "
        "temptation in a single round (part 1). It works through repetition: it makes "
        "the loss from falling into mutual betrayal larger, and so it increases the "
        "value of keeping the other player cooperating. The menu is small, so the result "
        "is a benchmark, not a complete analysis of the repeated game.")


def main():
    out("EQUILIBRIUM CHECKS FOR PART A")
    out("=" * 60)
    out("Nashpy: " + (f"version {nashpy.__version__}" if nashpy is not None
                      else "not installed (optional)"))
    out()
    part_one()
    part_two()
    path = settings.PROJECT_DIR / "results" / "equilibria_check.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
