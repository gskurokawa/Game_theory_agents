# Replication of the whole study with gpt-6-luna (run from the project folder in PowerShell).
# Before this, .env must contain MODEL=gpt-6-luna and RUN_SUFFIX=_gpt6luna.
# Data go to folders whose names end in _gpt6luna, so the gpt-5.6-luna data are not touched.
# Each command stops by itself at its cost limit (dollars, estimated). A command can be
# started again: games already finished are skipped.
# Run the commands one at a time, in this order, and check your OpenAI balance in between.

# 0. Trial: 2 games per condition, to check that the model accepts the settings (about $0.05).
python run_part_a.py --pilot --pilot-games 2 --max-cost 0.20

# 1. Baseline, prisoner's dilemma and public goods game (about $0.5).
python run_part_a.py --max-cost 1.00

# 2. Philanthropist and Christian/God, prisoner's dilemma AND public goods game (about $0.7 and $0.5).
python run_part_a.py --opponent-note philanthropist --max-cost 1.50
python run_part_a.py --opponent-note christian_god --max-cost 1.20

# 3. Same-model sentence, both games (about $0.5).
python run_part_a.py --opponent-note same_model --max-cost 1.00

# 4. Negative payoffs and threshold bonus, prisoner's dilemma only (about $0.2 and $0.3).
python run_part_a.py --game-types pd --pd-variant negative --max-cost 0.60
python run_part_a.py --game-types pd --prize --games 25 --max-cost 0.80

# 5. Part B (about $0.5).
python run_part_b.py --max-cost 1.20

# 6. Part B, public goods version: the nine conditions written for contributions (about $0.5).
python run_part_b.py --game pgg --pilot --pilot-games 2 --max-cost 0.20
python run_part_b.py --game pgg --max-cost 1.20
# Analysis: python analyze_part_b_pgg.py --data data\part_b_pgg_gpt6luna
