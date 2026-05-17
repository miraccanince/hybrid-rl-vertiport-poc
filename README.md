# Hybrid RL Safety Supervisor — Vertiport Arrival PoC

A lightweight proof-of-concept environment for testing a hybrid reinforcement
learning architecture applied to UAM vertiport arrival sequencing. The central
question it investigates is whether giving an RL agent specific reasons for
safety overrides, rather than a generic penalty, helps it learn a safer policy
faster.

This is not a complete arrival management system. It is a minimal, hand-crafted
simulation built to isolate and test one design decision before scaling to a
full simulator.

---

## Project Structure

```
toy-safe-uam/
├── env/
│   └── vertiport_env.py        Gymnasium environment and simulation physics
├── supervisor/
│   └── safety_supervisor.py    Rule-based layer that checks and corrects unsafe actions
├── training/
│   └── train_benchmark.py      PPO training script and comparison plot
├── results/
│   └── comparison.png          Generated after running the training script
└── requirements.txt
```

---

## How It Works

Three aircraft approach a single landing pad along a 1D corridor. A centralised
RL agent acts as a control tower, issuing speed commands for all aircraft each
timestep. A rule-based safety supervisor checks those commands before they are
executed and overrides any that would cause a violation.

**State space** — 10 dimensions:
- Distance, speed, and estimated time of arrival for each of the 3 aircraft
- Remaining pad occupation time

**Action space** — 27 discrete actions:
- Joint acceleration command for all 3 aircraft, each chosen from {-1, 0, +1}
- Encoded as a single integer and decoded via base-3 arithmetic

**Safety layer** — looks one step ahead:
- SEP: if two aircraft will be closer than the minimum separation threshold,
  the trailing aircraft is forced to decelerate
- PAD: if an aircraft will reach the pad while it is still occupied, it is
  forced to decelerate or hold

The physics engine always runs on the corrected commands, so safety is
guaranteed regardless of what the RL agent proposes.

---

## Mod A vs Mod B

This is the core experiment. Both modes use identical environments, agents,
and training budgets. The only difference is how the safety penalty is structured.

**Mod A (baseline hybrid RL)**
Any supervisor intervention triggers a flat penalty of -10. The agent learns
that some actions are unsafe, but receives no information about why.

**Mod B (intervention-aware hybrid RL)**
The penalty depends on which rule was violated:
- -15 for a separation breach (SEP), which carries higher physical risk
- -8 for a pad timing conflict (PAD), which is a scheduling error

The hypothesis is that structured feedback reduces repeated unsafe behaviour
and leads to faster, more stable convergence compared to the flat penalty
baseline.

---

## How to Run

```bash
pip install -r requirements.txt
python training/train_benchmark.py
```

Runs from the project root directory.

---

## Expected Results

The script trains both Mod A and Mod B PPO agents for 50,000 timesteps each
and saves a two-panel figure to `results/comparison.png`.

- Left panel: smoothed episode reward over training timesteps, showing how
  quickly each mode converges to an efficient landing policy
- Right panel: cumulative SEP and PAD intervention counts over time, showing
  how quickly each mode reduces repeated safety violations

---

## Relation to Thesis Work

This repository is a self-contained proof of concept prepared ahead of a
Master's thesis internship at Royal NLR on hybrid reinforcement learning for
vertiport arrival management. The full thesis will implement a similar
architecture in the BlueSky air traffic simulation environment, with additional
complexity including multi-route topology, energy modelling, and variable
traffic density.
