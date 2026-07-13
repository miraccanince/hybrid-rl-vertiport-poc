# Project Summary — Hybrid RL Safety Supervisor PoC

---

## 1. What real problem does this address?

Future Urban Air Mobility networks will require automated systems to sequence
hundreds of aircraft approaching vertiports simultaneously. Purely rule-based
systems scale and are safe, but they cannot adapt or optimise as traffic
conditions change. Purely RL-based systems can optimise, but unrestricted
exploration in safety-critical environments produces dangerous behaviour during
training.

The standard answer to this is a hybrid architecture: let the RL agent propose
actions, let a rule-based supervisor override unsafe ones. This exists in the
literature. The problem that is less addressed is what happens after the
override. The supervisor blocks the action and the agent receives a generic
penalty. The agent knows it made a mistake, but not why. It will likely make
the same class of mistake again.

This project investigates one specific design decision: does giving the RL
agent a structured reason for each override (violation type, which aircraft,
severity) help it learn a safer policy faster than a flat penalty signal?

---

## 2. What was the most critical architectural decision?

Keeping the safety supervisor as a fully standalone module rather than
embedding the safety logic inside the environment's step function.

The alternative would have been simpler to write initially — a few if-statements
inside step() that check distances and pad status. But this would have made
Mod A and Mod B impossible to compare cleanly, because the safety logic and
the reward logic would be tangled together.

By separating SafetySupervisor into its own class with a clean interface:

    safe_commands, log = supervisor.check(distances, speeds, tau_pad, active, commands)

the environment's step function only has one decision to make: if log is not
None, apply the right penalty based on mode. Everything else is identical
between Mod A and Mod B. This means the experiment is controlled — one
variable changes, everything else stays the same.

Second important decision: using Discrete(27) instead of MultiDiscrete([3,3,3])
for the action space. MultiDiscrete is more intuitive for a joint per-aircraft
command, but most Stable-Baselines3 algorithms only natively support Discrete
or Box. Using Discrete(27) with base-3 decoding keeps full SB3 compatibility
without a custom wrapper.

---

## 3. The hardest technical problems

**Problem 1 — Timestep ordering inside step().**
The tau_pad counter controls pad availability. Getting the order wrong produces
a subtle off-by-one bug: if you decrement tau_pad after checking for landings,
an aircraft that moves to d=0 on the step when tau_pad reaches 0 has to wait
an extra step before it can land. Physically, the pad clears during the same
tick that the aircraft arrives — so the decrement must happen before the
landing check, not after.

This kind of bug does not throw an error. It just produces slightly longer
episodes and slightly lower rewards, which are easy to misattribute to the
policy rather than the physics.

**Problem 2 — SB3 actual timesteps exceeding TOTAL_TIMESTEPS.**
PPO collects full rollouts of n_steps=2048 before updating. When
TOTAL_TIMESTEPS=50000 is not divisible by 2048, SB3 rounds up to the next
full rollout: 25 x 2048 = 51200. The callback accumulates 51200 entries in
sep_counts but the plot was generating a steps array of length 50000, causing
a shape mismatch crash at the very end of an otherwise successful training run.
Fix: derive the steps array from len(callback.sep_counts), not from
TOTAL_TIMESTEPS.

**Problem 3 — Observation values for landed aircraft.**
When an aircraft lands, it is retired from the simulation but the observation
vector still has slots for it. Setting speed to 0.0 for a landed aircraft
violates the declared observation space bounds (obs_low has V_MIN=1.0 as the
floor for speed), which causes Gymnasium's check_env to fail. The fix is to
use V_MIN as a sentinel value for landed aircraft speed — it is technically in
bounds and semantically neutral, since the aircraft is no longer moving.

---

## 4. How the system works

```
Episode start
    |
    v
reset() -- randomise positions (staggered windows) and speeds
    |
    v
Agent observes state (10-dim vector: d, v, ETA per aircraft + tau_pad)
    |
    v
Agent proposes joint action (integer 0-26)
    |
    v
_decode_action() -- base-3 decoding to per-aircraft commands [-1, 0, +1]
    |
    v
SafetySupervisor.check() -- predict next state under proposed commands
    |-- PAD violation? force command to -1 (or 0 if at V_MIN)
    |-- SEP violation? force trailing aircraft command to -1
    |-- No violation? pass through unchanged
    |
    v
Physics update -- speeds clipped, distances decremented
    |
    v
Landing detection -- d==0 and tau_pad==0 -> retire aircraft, set tau_pad=TAU_LAND
    |
    v
Reward = R_STEP + landings * R_LAND + penalty(log, mode)
    Mod A: any intervention -> -10
    Mod B: SEP -> -15,  PAD -> -8
    |
    v
info dict populated with intervention telemetry for callback logging
    |
    v
Repeat until all aircraft landed or MAX_STEPS reached
```

---

## 5. Results and what they mean

After 50,000 PPO timesteps:

- Both Mod A and Mod B converge to roughly the same final episode reward (~+35).
  This is expected — the environment is small and PPO is capable enough to find
  a reasonable policy regardless of reward structure.

- Mod B's SEP curve flattens slightly earlier than Mod A. The agent receiving
  a harsher SEP penalty learns to avoid separation violations faster.

- Mod B accumulates more PAD interventions than Mod A. This is not a failure.
  Mod B's softer PAD penalty (-8 vs -10 in Mod A) means the agent rationally
  tolerates more pad timing errors in exchange for avoiding the more costly SEP
  violations. It learned the relative severity ordering between the two violation
  types.

This PAD/SEP trade-off is the most interesting result in the experiment. A flat
penalty (Mod A) gives the agent no basis to prioritise one violation type over
another. A structured penalty (Mod B) implicitly encodes a priority ordering
that the agent internalises. In a real arrival management system this maps
directly to the distinction between safety-critical conflicts and
efficiency-level scheduling errors.

---

## 6. What needs to happen before scaling to a real simulator

This PoC deliberately simplifies everything that is not the core research
question. Scaling to the BlueSky thesis environment would require:

Critical:
- 2D or 3D airspace geometry instead of a 1D corridor
- Multiple arrival routes and route conflict detection
- Variable traffic density (the PoC always has exactly 3 aircraft)
- Real aircraft performance envelopes (speed limits depend on aircraft type
  and phase of flight, not a fixed [1, 5] range)

Important:
- Energy and battery state modelling (relevant for eVTOL aircraft)
- Multi-pad vertiport layout
- Wind and weather effects on ETA predictions
- Evaluation against a rule-based baseline (FCFS) to show RL adds value

Research:
- The Mod A vs Mod B comparison needs more seeds and longer training runs
  to produce statistically significant results
- The intervention feedback mechanism should be tested with other algorithms
  (SAC, TD3) to check whether the effect is PPO-specific
- The structured penalty is one way to feed intervention information back.
  Others (observation augmentation, replay buffer weighting) remain untested.

---

## 7. The most important thing learned from this project

Technically: separating the safety supervisor from the environment was the
right call, but the reason it matters is not just code cleanliness. It forced
a precise answer to the question "what exactly is the supervisor responsible
for?" — predicting violations one step ahead and correcting commands, nothing
else. Reward shaping is the environment's responsibility. Logging is the
callback's responsibility. When each component has one job, bugs are easier
to find and the experiment is easier to trust.

On the research question: the result is promising but modest. Mod B learns a
slightly safer separation policy faster, but the overall reward difference is
small at 50,000 timesteps. This is actually the honest result for a toy
environment — the effect is likely to be larger in a more complex simulator
where the agent has more opportunities to make the same mistake repeatedly and
where the consequences of different violation types are more differentiated.
The PoC shows the mechanism works. Whether it works at scale is the thesis.
