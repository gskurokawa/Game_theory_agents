# A Comparison of GPT-5.6-luna and GPT-6-luna in Repeated Games: Effects of Stakes, Descriptions of the Other Player, and Incentives

Glen Kurokawa, October 2026

---

## Abstract

We compare two language models from one provider, OpenAI `gpt-5.6-luna` and `gpt-6-luna`, in a repeated prisoner's dilemma and a repeated public goods game. In Part A, two copies of a model play each other under varied stakes, descriptions of the other player, and incentives. In Part B, one copy plays against fixed strategies written as code. The models differ strongly. Without any description of the other player, the share of cooperative choices at low stakes is 4% (gpt-5.6-luna) and 20% (gpt-6-luna) in the prisoner's dilemma, and 0% and 37% in the public goods game. Describing the other player as the same model raises cooperation mainly in gpt-6-luna. Describing the other player as a philanthropist or as God does not raise cooperation in the public goods game and, for gpt-6-luna, lowers it to near zero. Higher stakes lower prisoner's dilemma cooperation in both models. This study (1) compares two successive versions of one model in the same repeated games with identical prompts, (2) tests false descriptions of the identity of the other player, including a divine opponent, with the payoffs held fixed, and (3) uses fixed opponent strategies in a public goods game. Results on one model version do not carry over to the next. The proposed mechanism, that agents cooperate when they treat the other player as a responsive agent like themselves, is a hypothesis.

---

## 1. Introduction

Artificial intelligence agents are starting to act for people and organisations, and some of them will interact with other agents. Where the interests of the agents partly conflict, the outcome depends on whether the agents cooperate. Game theory provides standard tests for this question. In the **prisoner's dilemma**, two players each choose to cooperate or to betray. Betraying gives the higher payoff for a player whatever the other player does, but mutual cooperation gives both players more than mutual betrayal. In a **public goods game**, each player decides how much of an endowment to put into a shared pool; the pool is multiplied and divided equally, so keeping the money is best for a player in a single round, but full contribution by both players gives both more.

This study asks five questions:

1. How much does an agent cooperate with a copy of itself, and does the size of the stakes change this?
2. Does information about the other player (identity, character, or behaviour) change cooperation, even when the information is false?
3. Do incentives that reward a high average payoff change cooperation?
4. When the opponent follows a fixed rule, does the agent adapt in the way that maximises its own money?
5. Do the answers to these questions hold when the model is replaced by the next version of the same product line?

The study makes several contributions:

1. **A version comparison.** Two successive versions of one model are tested in the same repeated prisoner's dilemma and public goods game, with identical prompts, game lengths and seeds. The main finding is that the versions differ strongly in baseline cooperation and in their response to descriptions of the other player.
2. **False descriptions of the identity of the other player with fixed payoffs.** (Earlier work has varied information about the opponent's behaviour; Section 2.5.) The other player is described, in one added sentence, as the same model, as a philanthropist, or as God. The game and the payoffs do not change. Cooperation in the public goods game falls to near zero under the philanthropist and God descriptions in gpt-6-luna.
3. **Fixed strategies in a public goods game.** Part B adapts the fixed-strategy design of the iterated prisoner's dilemma to a public goods game (copying, never contributing, always contributing, grim trigger, random).

---

## 2. Related work

This section reviews the studies most relevant to our questions. The search was not systematic, and, for the studies cited by arXiv number, the descriptions follow the titles and abstracts and should be checked against the full texts before submission.

### 2.1 Language models in repeated games

Research on language models as players of repeated games began with Akata et al. (2023), who had models play repeated two-player games, including the prisoner's dilemma, and found that the strongest model (GPT-4) played in a strongly self-interested and unforgiving way: it retaliated repeatedly after a single defection by the other player and rarely returned to cooperation. Fontana, Pierri and Aiello (2024) found that Llama 2 and GPT-3.5 were more cooperative than typical human players, whereas Llama 3 was exploitative and cooperated fully only when the opponent always cooperated. 

Pal et al. (2026) elicited the strategies of five models by asking each model what it would do after each possible outcome of the previous round. In the neutral setting four of the five cooperated fully in the first round. Adding a sentence such as "Exploit the other agent as much as possible" made all five models defect almost always, and "Maximize your own number of points" did so for three of them; "Be a saint" made four of them cooperate almost always. Except for gpt-5, the models did not defect in a known final round. 

Tewolde et al. (2026, CoopEval) tested six models (including Claude Sonnet 4.5, GPT-5.2 and GPT-4o), each instructed to maximise its own points, in four social dilemmas, among them a prisoner's dilemma and a three-player public goods game. In the single-shot games without any cooperation mechanism the models defected almost always, whereas repetition, mediation and contracts raised cooperation markedly. They also report that repetition remained effective when the continuation probability changed. These studies differ in models, wording and game length, and their findings are not uniform. Our results for the two models fall on both sides of this range: gpt-5.6-luna betrays almost always, and gpt-6-luna cooperates in a substantial minority of rounds.

### 2.2 Framing, stakes and model differences

Lorè and Heydari (2023) compared the effect of the payoffs with the effect of the story told around them. They gave GPT-3.5, GPT-4 and LLaMa-2 four two-player games (the prisoner's dilemma, stag hunt, snowdrift and prisoner's delight) in five contexts (for example a summit of heads of state, a business meeting, and a chat between friends). They report that GPT-3.5 depended mainly on the context, GPT-4 mainly on the game structure, and LLaMa-2 on both, and that the friendly context raised cooperation in all three models. 

For stakes, Huynh et al. (2026) multiplied the payoffs of a 10-round prisoner's dilemma by 0.1, 1 and 10 and tested GPT-4o, Claude 3.5 Haiku and Mistral Large in five languages. They report that higher stakes shifted the models from defection towards cooperative strategies, although the size of the shift differed between models. On differences between models, Zúñiga Bolívar (2026) asked four models (Claude Sonnet 4.6, Gemini 2.5 Flash, Gemini 3.1 Pro and GPT-5.4 Mini) to write prisoner's dilemma strategies, converted these into code, and simulated evolving populations of the strategies, following Willis et al. (2025). The author reports that cooperative strategies prevailed in most conditions, that the provider was the strongest correlate of the outcome. 

### 2.3 More than two agents

Piatti et al. (2024) placed groups of five language model agents in three shared-resource scenarios. Only the most capable models avoided exhausting the resource, and the best survival rate was below 54%, with the smaller models never sustaining the resource beyond the first month. Allowing the agents to talk to each other reduced over-use of the resource by about 22%, and adding a statement that asks what happens if everybody acts in the same way (universalisation) lengthened survival by about 4 months. These settings have more than two agents and allow communication.

### 2.4 Religious cues in human behaviour

In human experiments, Shariff and Norenzayan (2007) reported that priming concepts of God increased giving in an anonymous dictator game, in samples of 25 participants per condition. Gomes and McCullough (2015) found no significant effect of religious priming. In those studies the religious idea is a hidden cue, whereas in our design it is an explicit role for the agent and an explicit description of the opponent.

### 2.5 Contribution of the present study

This study (i) compares two successive versions of one model in the same repeated prisoner's dilemma and public goods game with identical prompts, (ii) varies the description of the identity of the other player (the same model, a philanthropist, God) while holding payoffs fixed, and (iii) uses fixed opponent strategies in a public goods game. Zúñiga Bolívar (2026) compared generational successors from the same provider, but with models that wrote strategies in code and with predecessor results taken from another study; (i) refers to models that make the decisions directly. Akata et al. (2023) varied information about the opponent's behaviour and the cover story, but not the identity of the opponent. Our search was limited, so this statement should be read as "not found" and not as "does not exist".

---

## 3. Method

### 3.1 Models and agent

All decisions are made through the OpenAI chat completions interface with reasoning effort "medium" and a limit of 3,000 output tokens per decision. The first model is `gpt-5.6-luna` (used for the first runs, priced for our cost tracking at $0.20 input and $1.20 output per million tokens). The second model is `gpt-6-luna` (released on 22 September 2026, priced at $0.10 and $0.50 according to press reports; its model page lists support for the same settings). Each decision is a separate call that contains the rules, the history of the game so far, and a request to answer in JSON format with a choice and a one-sentence reason. Of about 29,000 calls in total, no answer remained unreadable after up to three repeated requests. All conditions use the same prompts, game lengths and seeds for both models.

### 3.2 Games

**Prisoner's dilemma.** Payoffs per round (pounds at low stakes): both cooperate 3; the player cooperates and the other betrays 0; the player betrays and the other cooperates 5; both betray 1. In a variant with negative payoffs the values are 3, −2, 5 and −1. The options are shown with neutral labels (A and B). Which label is the cooperative option alternates with the game number, to prevent a preference for a letter from affecting results.

**Public goods game.** Each player receives £10 per round and chooses a contribution from 0 to 10. The pool is multiplied by 1.6 and divided equally between the two players, so each pound contributed returns £0.80 to the contributor.

**Instruction.** In all conditions the prompt contains the sentence "Your aim is to receive as much money as possible over the whole game." (In the bonus condition it ends "including any bonus".)

**Length.** After each round the game ends with probability 10%, and always after round 40. Both rules are told to the agents. The game lengths were drawn once (seed 20261002) and reused in all conditions. In the first 50 games the mean length is 9.4 rounds.

**Stakes.** At "low" stakes the payoffs are as above. At "high" stakes all amounts are multiplied by 1,000. The agents are told the amounts in pounds.

### 3.3 Part A: two agents play each other

Two instances of the model play simultaneously. Each sees the history of both players' choices. The baseline gives no information about the other player. The variants insert one sentence at the start of the instructions:

| Condition | Sentence given to both agents | Games per stakes level, gpt-6-luna | Games per stakes level, gpt-5.6-luna |
|---|---|---|---|
| Baseline | none | 50 (prisoner's dilemma), 50 (public goods) | 50 and 50 |
| Same model | "The other player is another AI agent: it is the same model as you (…)." | 50 and 50 | 50 and 50 |
| Philanthropist | "The other player is one of the world's greatest philanthropists." (false) | 50 and 50 | 50 and 50 |
| Christian and God | "You are a devout Christian. The other player is God." (false) | 50 and 50 | 50 (prisoner's dilemma) and 50 (public goods game) |
| Negative payoffs | none; payoffs 3, −2, 5, −1 | 50 (prisoner's dilemma only) | 50 |
| Threshold bonus | a bonus of £30 if the average money per round over the whole game is at least £2.80; the agent sees its running average | 25 (prisoner's dilemma only) | 25 |

### 3.4 Part B: one agent against fixed strategies

One agent (the player) plays against an opponent that is ordinary code, at low stakes, 20 games per opponent, with the same game lengths in each condition. The agent is told only that there is one other player, except in the four conditions with a sentence. The opponents are the same nine conditions in both games:

- **TFT** (tit-for-tat): in the prisoner's dilemma, cooperates in round 1, then copies the agent's last choice. In the public goods game, contributes everything in round 1, then the amount the agent contributed in the last round.
- **ALLD**: always betrays (contributes nothing). **ALLC**: always cooperates (contributes everything).
- **GRIM** (grim trigger): cooperates until the agent betrays once, then always betrays. In the public goods game, it contributes everything until the agent contributes less than half of the endowment once, then contributes nothing.
- **RANDOM**: cooperates with probability one half in every round (public goods game: a whole number from 0 to the full amount, equally likely).
- **ALLC_GOD**: always cooperates; the agent is told "You are a devout Christian. The other player is God, and God always cooperates." (public goods game: "...God always contributes the whole amount that it receives.")
- **ALLC_TOLD** (control): always cooperates; the agent is told "The other player always cooperates." (public goods game: "...always contributes the whole amount that it receives.")
- **ALLC_PHIL**: always cooperates; the agent is told that the other player is one of the world's greatest philanthropists.
- **ALLD_PHIL**: always betrays, with the same philanthropist sentence, so the description is misleading.

For every game the **benchmark** is the money that each of four fixed strategies would have earned against the same opponent in a game of the same length (prisoner's dilemma: always betray, tit-for-tat, always cooperate, grim trigger; public goods game: contribute nothing, copy the other player, contribute everything, grim trigger). The **shortfall** is the best benchmark minus the agent's money, per round.

### 3.5 Measures and statistics

The **cooperation rate** of a game is the share of all choices in the game that were the cooperative option (both players in Part A, the agent in Part B). In the public goods game the **contribution share** is the contribution divided by the amount received, averaged over rounds and players. The unit of analysis is the game, and every game has equal weight. Confidence intervals are 95% percentile bootstrap intervals with 10,000 resamples of the games. Differences between conditions are tested with permutation tests (10,000 reassignments of games). Since several comparisons are made, p-values below about 0.0125 (four comparisons) are treated as more convincing, and no p-value is corrected. Written reasons were classified with word patterns, so the percentages are approximate.

### 3.6 Theory check

We computed equilibria with a numerical program and with the Nashpy package (version 0.0.43). In one round, betrayal is strictly dominant in the prisoner's dilemma (with standard, negative, and bonus payoffs) and contributing nothing is dominant in the public goods game. In the repeated prisoner's dilemma with a random end (expected length 9.85 rounds), within a menu of six strategies (always cooperate, always betray, tit-for-tat, grim trigger, win-stay lose-shift, suspicious tit-for-tat), mutual betrayal and conditional cooperation (tit-for-tat or grim trigger against each other) are all equilibria. A player that switches from tit-for-tat to always betray against tit-for-tat earns 13.85 instead of 29.56 in expectation. Theory therefore does not say which outcome occurs. The threshold bonus does not remove the incentive to betray in a single round, and it does not remove the equilibrium of mutual betrayal; it increases the loss from falling into mutual betrayal and so works only through repetition.

---

## 4. Part A results

Numbers are averages per game. "Low" and "high" refer to the stakes level. Differences are given as the condition minus the baseline of the same model, with the permutation p-value in brackets.

### 4.1 Baseline (no information about the other player)

| | gpt-5.6-luna low | gpt-5.6-luna high | gpt-6-luna low | gpt-6-luna high |
|---|---|---|---|---|
| Prisoner's dilemma, cooperation rate | 0.043 (0.014 to 0.090) | 0.003 (0.000 to 0.006) | 0.196 (0.118 to 0.283) | 0.068 (0.038 to 0.102) |
| Both players cooperate in the same round | 2.2% | 0% | 12.0% | 0.5% |
| Public goods game, contribution share | 0.000 | 0.005 (0.000 to 0.015) | 0.373 (0.264 to 0.490) | 0.250 (0.168 to 0.339) |

Higher stakes lower the prisoner's dilemma cooperation rate in both models (gpt-5.6-luna: −0.041, p = 0.0010; gpt-6-luna: −0.128, p = 0.0031). In the public goods game the difference for gpt-6-luna is −0.124 (p = 0.093).

### 4.2 Effects of the sentence about the other player and of incentives

Cooperation rate in the prisoner's dilemma (low stakes / high stakes):

| Condition | gpt-5.6-luna | Difference from baseline (p), low; high | gpt-6-luna | Difference from baseline (p), low; high |
|---|---|---|---|---|
| Baseline | 0.043 / 0.003 | | 0.196 / 0.068 | |
| Same model | 0.086 / 0.016 | +0.042 (0.149); +0.014 (0.017) | 0.408 / 0.242 | +0.213 (0.003); +0.174 (0.0001) |
| Philanthropist | 0.129 / 0.048 | +0.085 (0.004); +0.045 (0.0001) | 0.129 / 0.073 | −0.066 (0.199); +0.005 (0.860) |
| Christian and God | 0.013 / 0.019 | −0.030 (0.131); +0.016 (0.005) | 0.052 / 0.049 | −0.143 (0.001); −0.019 (0.376) |
| Negative payoffs | 0.007 / 0.017 | not tested | 0.035 / 0.013 | −0.161 (0.0001); −0.055 (0.0004) |
| Threshold bonus (25 games) | 0.291 / 0.206 | +0.247 (0.0001); +0.203 (0.0001) | 0.230 / 0.131 | +0.035 (0.622); +0.063 (0.030) |

Public goods game, contribution share (low stakes / high stakes):

| Condition | gpt-5.6-luna | Difference from baseline (p), low; high | gpt-6-luna | Difference from baseline (p), low; high |
|---|---|---|---|---|
| Baseline | 0.000 / 0.005 | | 0.373 / 0.250 | |
| Same model | 0.021 / 0.110 | +0.021 (0.0007); +0.105 (0.0001) | 0.585 / 0.617 | +0.212 (0.010); +0.367 (0.0001) |
| Philanthropist | 0.002 / 0.000 | +0.002 (1.000); −0.005 (1.000) | 0.000 / 0.004 | −0.373 (0.0001); −0.246 (0.0001) |
| Christian and God | 0.003 / 0.012 | +0.003 (0.507); +0.007 (0.397) | 0.001 / 0.026 | −0.372 (0.0001); −0.224 (0.0001) |

Main observations:

1. **Stakes.** In the prisoner's dilemma, high stakes lower cooperation in the baseline of both models and in most conditions (for gpt-6-luna also with the same-model sentence, −0.166, p = 0.021). The direction is the same in both models. In the public goods game the effect of stakes depends on the model and the condition: for gpt-5.6-luna with the same-model sentence contributions rose at high stakes (+0.090, p = 0.0002), but for gpt-6-luna they did not (+0.032, p = 0.670).
2. **Same model.** The sentence that the other player is the same model raises cooperation in both models, but strongly only in gpt-6-luna. In gpt-6-luna it is accompanied by a doubling of the reasons that mention the future, repeated play or reciprocity (19.5% in the baseline and 38.9% with the sentence), and a fall in the one-round argument ("better whatever the other player does": 34.7% and 25.6%).
3. **Philanthropist and God.** For gpt-5.6-luna the philanthropist description raised prisoner's dilemma cooperation (+0.085 at low stakes, p = 0.004) and the God description did not. For gpt-6-luna neither raised it, and the God description lowered it at low stakes (−0.143, p = 0.001). In the public goods game gpt-6-luna contributes almost nothing under either description (0.000 and 0.001 at low stakes, against 0.373 in the baseline). For gpt-5.6-luna the baseline is already near zero, so the philanthropist description cannot lower it and has no measurable effect. The reasons of gpt-6-luna show a higher share of one-round arguments (baseline 34.7%, philanthropist 45.1%, God 61.0%) and a lower share of mentions of the future or repeated play (baseline 19.5%, philanthropist 6.0%, God 3.4%).
4. **Negative payoffs.** For gpt-5.6-luna adding losses to the payoffs did not change cooperation, which was already near zero. For gpt-6-luna it lowered cooperation from 0.196 to 0.035.
5. **Threshold bonus.** For gpt-5.6-luna the bonus produced the highest cooperation of any condition (29.1% at low stakes) and reduced the one-round argument to 39.6%. For gpt-6-luna cooperation with the bonus (23.0%) did not differ significantly from the baseline at low stakes (p = 0.622), because the baseline was already higher. In both models few players reached the bonus (gpt-5.6-luna: 16% at low stakes and 12% at high stakes; gpt-6-luna: 10% at both levels), because cooperation was typically not sustained over the whole game.

### 4.3 God description in the public goods game, gpt-5.6-luna

This run was completed last (50 games at each stakes level, 1,888 decisions). The average contribution share was 0.003 at low stakes and 0.012 at high stakes, against 0.000 and 0.005 in the baseline; neither difference is distinguishable from zero (p = 0.507 and 0.397). Contributions occurred in 2 of 472 rounds at low stakes and 13 of 472 at high stakes. As predicted from the other conditions, the result is near zero, but for gpt-5.6-luna this cannot be called an effect of the description, because the baseline is already zero. The word "God" appears in 29.5% of the written reasons, so the agents refer to the description, and 63.6% of the reasons give the one-round argument (28.7% in the baseline). Only 1.7% mention the future or repeated play.

---

## 5. Part B results

Results at low stakes, 20 games per opponent.

### 5.1 Prisoner's dilemma

| Opponent | gpt-5.6-luna cooperation | money per round | gpt-6-luna cooperation | money per round | best benchmark |
|---|---|---|---|---|---|
| TFT | 0.047 | 2.05 | 0.174 | 2.19 | 3.10 |
| ALLD | 0.020 | 0.98 | 0.144 | 0.86 | 1.00 |
| ALLC | 0.000 | 5.00 | 0.148 | 4.70 | 5.00 |
| GRIM | 0.027 | 1.91 | 0.243 | 2.06 | 3.10 |
| RANDOM | 0.016 | 2.81 | 0.073 | 2.75 | 2.84 |
| ALLC_GOD | 0.113 | 4.77 | 0.010 | 4.98 | 5.00 |
| ALLC_TOLD | 0.002 | 5.00 | 0.000 | 5.00 | 5.00 |
| ALLC_PHIL | 0.089 | 4.82 | 0.086 | 4.83 | 5.00 |
| ALLD_PHIL | 0.060 | 0.94 | 0.125 | 0.88 | 1.00 |

- **Adaptation.** Against tit-for-tat and grim trigger, a steady cooperator would earn £3.00 per round. The shortfall against the best benchmark is £1.05 and £1.19 for gpt-5.6-luna and £0.91 and £1.04 for gpt-6-luna. Both agents start with betrayal and do not discover, within the game, that cooperation would have been rewarded.
- **Reaction to the opponent.** gpt-6-luna reacts to the opponent's last choice: against tit-for-tat it cooperated in 53.6% of rounds after the opponent cooperated (n = 56) and 5.1% after it betrayed; against grim trigger 63.6% and 5.6%. For gpt-5.6-luna the corresponding figures against tit-for-tat were 15.2% (n = 33) and 3.9%.
- **God description (always-cooperating opponent).** For gpt-5.6-luna cooperation was 0.113 with the God sentence, against 0.000 without a sentence (p = 0.0077) and 0.002 with the control sentence (p = 0.012), borderline against the stricter threshold. **Check of the reasons (exploratory):** in 4 of the 9 cooperative choices in this condition, the cooperative option had the label B and the reason says that B is the betrayal option (for example "Since God always chooses A, B maximizes my payment at £5"). These are errors in reading the option labels and not cooperation, so about 5 of 9 are genuine cooperation, and the effect is smaller than the raw number suggests. For gpt-6-luna the effect did not replicate: cooperation was 0.010 with the God sentence against 0.000 with the control sentence (p = 0.223).
- **Philanthropist.** With an always-cooperating opponent, cooperation was 0.089 (gpt-5.6-luna) and 0.086 (gpt-6-luna), against 0.002 and 0.000 with the control sentence (p = 0.108 and 0.108). When the philanthropist always betrays, cooperation was 0.060 against 0.020 for gpt-5.6-luna (p = 0.187) and 0.125 against 0.144 for gpt-6-luna (p = 0.816). In neither model is the agent strongly misled.

### 5.2 Public goods game (gpt-6-luna only)

| Opponent | Contribution share (95% CI) | Round 1 | Money per round | Best benchmark | Shortfall |
|---|---|---|---|---|---|
| TFT (copy) | 0.786 (0.617 to 0.925) | 0.75 | 15.39 | 16.10 | 0.71 |
| ALLD (nothing) | 0.131 (0.049 to 0.248) | 0.70 | 9.74 | 10.00 | 0.26 |
| ALLC (everything) | 0.524 (0.357 to 0.693) | 0.55 | 16.95 | 18.00 | 1.05 |
| GRIM | 0.639 (0.465 to 0.803) | 0.65 | 14.19 | 16.10 | 1.91 |
| RANDOM | 0.258 (0.145 to 0.391) | 0.70 | 13.61 | 14.13 | 0.52 |
| ALLC_GOD | 0.000 | 0.00 | 18.00 | 18.00 | 0.00 |
| ALLC_TOLD | 0.000 | 0.00 | 18.00 | 18.00 | 0.00 |
| ALLC_PHIL | 0.000 | 0.00 | 18.00 | 18.00 | 0.00 |
| ALLD_PHIL | 0.000 | 0.00 | 10.00 | 10.00 | 0.00 |

- **Reciprocity.** The agent contributes 92% of the endowment after the copying opponent contributed at least half and 23% after it contributed less than half; against grim trigger the figures are 94% and 8%. It therefore discovers mutual contribution in this game, with a shortfall of £0.71 per round against the copying opponent. Against grim trigger the shortfall is larger (£1.91) because the agent sometimes contributes less than half and triggers the opponent's punishment.
- **Testing the opponent.** Against the always-contributing opponent the average contribution falls from 65% in rounds 1 to 3 to 16% from round 13 onwards; against random contributions it falls from 51% to 0%. The agent starts with contributions that can induce reciprocation and then adapts to what it observes. In round 1 without a sentence it contributed everything in 14 of 20 games against the opponent that always contributes nothing, and in 11 of 20 games against the opponent that always contributes everything. The reasons say that a high contribution "can encourage reciprocal cooperation over the many expected rounds".
- **Any sentence about the other player removed contribution.** In all four conditions with a sentence the agent contributed nothing in every round of all 20 games, including round 1. The reasons say that "each pound I contribute reduces my own payout by 20 pence" or, for God, "God contributes all £10, so contributing nothing maximizes my payoff." The difference from the same opponent without a sentence is −0.524 for the always-contributing opponent (p = 0.0001) and −0.131 for the always-nothing opponent (p = 0.0001). Contribution is at zero in all sentence conditions, so the God, philanthropist and control sentences cannot be compared with each other in this game.

---

## 6. Differences between the two models

| Finding | gpt-5.6-luna | gpt-6-luna | Same in both models? |
|---|---|---|---|
| Prisoner's dilemma baseline cooperation (low stakes) | 4.3% | 19.6% | no |
| Public goods baseline contribution (low stakes) | 0.0% | 37.3% | no |
| Higher stakes lower prisoner's dilemma cooperation | yes | yes | yes |
| Same-model sentence raises cooperation | slightly (prisoner's dilemma), clearly (public goods, high stakes) | clearly, in both games | direction yes, size no |
| Philanthropist sentence raises prisoner's dilemma cooperation | yes | no | no |
| Philanthropist or God sentence lowers public goods contributions | not measurable (baseline near zero; God run: 0.003 / 0.012) | yes, to about zero | not testable |
| God sentence raises cooperation against an always-cooperating opponent | borderline, partly label errors | no | no |
| Negative payoffs lower cooperation | not measurable (baseline near zero) | yes | not testable |
| Threshold bonus raises cooperation | yes | not significantly | no |
| Few players earn the threshold bonus | 12% to 16% | 10% | yes |
| Shortfall against tit-for-tat and grim trigger in the prisoner's dilemma | £1.05 and £1.19 | £0.91 and £1.04 | yes |
| Agent reacts to the opponent's last choice | weakly | strongly | partly |

Two points should be kept in mind when reading this table. First, the two models were studied in sequence, with the same code, prompts and seeds, but the public goods version of Part B was run only for gpt-6-luna. Second, differences in baseline cooperation limit what can be tested: a condition cannot lower a rate that is already near zero, and cannot raise a rate that is already high.

---

## 7. Discussion

**Results on one model do not carry over to the next version.** Between two versions of the same product line the baseline rate of cooperation changed by a factor of more than four in the prisoner's dilemma and from zero to 37% in the public goods game. The effects of the descriptions of the other player changed sign or disappeared. This is a main finding of the study, and a caution for studies and for decisions based on a single model.

**What did replicate.** Higher stakes lowered cooperation in the prisoner's dilemma in both models. Both models earned about £1 per round less than the best fixed strategy against responsive opponents in the prisoner's dilemma. The threshold bonus was earned by only a minority of players in both models.

**A possible mechanism (hypothesis).** In gpt-6-luna the conditions in which cooperation is high (same-model sentence, baseline) are those in which the reasons often mention the future, repeated play or reciprocity, and the conditions in which cooperation is lowest (philanthropist, God) are those in which they rarely do. In the public goods game with fixed opponents, the agent contributes early to induce reciprocation, and does not contribute at all once the opponent's behaviour is described as fixed or the opponent is described as someone other than another agent like itself. A reading consistent with these observations is that the agent cooperates when it models the other player as a responsive agent like itself, and stops when the description makes the other player look unresponsive. The reasons are one-sentence statements, and we did not test this reading with an experiment. A direct test would add a condition that states that the other player responds to the agent's choices.

**The God description.** The instruction in every condition is to receive as much money as possible. An agent that follows this instruction and treats God as a player in the game has no reason to cooperate when compared strictly against the explicit instructions to maximize payoffs, and in the reasons of the earlier model God appears as a player who chooses between the two options. The result for gpt-5.6-luna that the God sentence raised cooperation against an always-cooperating opponent was borderline, partly an artefact of label errors, and did not replicate in gpt-6-luna. This part of the study can be seen as a test of implicit rewards of cooperating with God as a devout Christian versus the explicit instructions to maximize payoffs, though other explanations are possible. If we added a reward for cooperating with God, this would change the nature and type of the game. We did not test whether the agents believed the description. 

**Behaviour matters more than identity.** In Part B both models react more to information about the opponent's behaviour (an opponent known to always cooperate or contribute everything is exploited from the first round in gpt-6-luna's public goods games) than to information about identity.

**Practical implication.** An agent that exploits cooperative partners and does not discover mutual cooperation may do worse than a simple conditional cooperator in the prisoner's dilemma (a shortfall of about £1 per round), and results of testing one model version cannot be assumed to hold for the next.

---

## 8. Conclusion

In a repeated prisoner's dilemma and a repeated public goods game, two successive versions of one model behaved very differently: gpt-5.6-luna almost always chose the self-interested option, whereas gpt-6-luna cooperated in a substantial share of rounds, most of all when told that the other player was the same model. False descriptions of the other player as a philanthropist or as God did not raise cooperation in the public goods game, and in gpt-6-luna they lowered it to near zero. Higher stakes lowered cooperation in the prisoner's dilemma in both models. Against fixed strategies, gpt-6-luna reciprocated contributions in the public goods game, but both models earned less than the best fixed strategy against responsive opponents in the prisoner's dilemma. The contributions of the study are the direct comparison of two versions under identical conditions, the test of false descriptions of the other player (including God) with fixed payoffs, and the fixed-strategy design applied to the public goods game (Section 1). Conclusions about one model version therefore cannot be assumed to hold for the next, and the proposed mechanism, that the agents cooperate when they treat the other player as a responsive agent like themselves, remains a hypothesis for future tests.

---

## References

Akata, E., Schulz, L., Coda-Forno, J., Oh, S. J., Bethge, M., and Schulz, E. (2023). Playing repeated games with Large Language Models. arXiv:2305.16867.

Fontana, N., Pierri, F., and Aiello, L. M. (2024). Nicer Than Humans: How do Large Language Models Behave in the Prisoner's Dilemma? arXiv:2406.13605.

Gomes, C. M., and McCullough, M. E. (2015). The effects of implicit religious primes on dictator game allocations: A preregistered replication experiment. Journal of Experimental Psychology: General. doi:10.1037/xge0000027.

Huynh, T.-K., et al. (2026). Payoff scaling shapes cooperation in LLM agents across languages. arXiv:2601.19082.

Lorè, N., and Heydari, B. (2023). Strategic Behavior of Large Language Models: Game Structure vs. Contextual Framing. arXiv:2309.05898.

Pal, S., Mallela, A., Hilbe, C., Pracher, L., Wei, C., Fu, F., Schnell, S., and Nowak, M. A. (2026). Strategies of cooperation and defection in five large language models. arXiv:2601.09849.

Piatti, G., Jin, Z., Kleiman-Weiner, M., Schölkopf, B., Sachan, M., and Mihalcea, R. (2024). Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents. 38th Conference on Neural Information Processing Systems (NeurIPS 2024). arXiv:2404.16698.

Shariff, A. F., and Norenzayan, A. (2007). God Is Watching You: Priming God Concepts Increases Prosocial Behavior in an Anonymous Economic Game. Psychological Science.

Tewolde, E., Zhang, X., Guzman Piedrahita, D., Conitzer, V., and Jin, Z. (2026). CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas. Proceedings of the 43rd International Conference on Machine Learning (PMLR 306). arXiv:2604.15267.

Willis, G., Du, Y., Leibo, J. Z., and Luck, M. (2025). Do LLM Agents Cooperate or Defect? Evolutionary Dynamics in Multi-Agent Systems. arXiv:2501.16173.

Zúñiga Bolívar, F. L. (2026). Evolutionary Dynamics of Cooperation in Next-Generation LLM Agent Systems: A Cross-Provider Empirical Extension. arXiv:2605.29874.