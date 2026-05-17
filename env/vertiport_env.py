"""
Toy vertiport arrival environment for a hybrid RL + safety supervisor PoC.
N aircraft approach a single landing pad along a 1D corridor. A centralised
agent (control tower) issues joint speed commands each timestep.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from supervisor.safety_supervisor import SafetySupervisor, InterventionLog


class VertiportEnv(gym.Env):
    """
    Single-pad vertiport arrival management environment.

    The agent controls all aircraft simultaneously (centralised control tower).
    Supports two supervision modes:
        mod_a  –  generic penalty on any safety intervention
        mod_b  –  reason-coded penalty (SEP vs PAD), giving the agent
                  structured feedback about why an action was blocked
    """

    metadata = {"render_modes": []}

    def __init__(self, n_aircraft: int = 3, mode: str = "mod_a") -> None:
        """
        Args:
            n_aircraft: number of aircraft (fixed at 3 for this PoC)
            mode: 'mod_a' for standard hybrid RL, 'mod_b' for intervention-aware
        """
        super().__init__()

        assert mode in ("mod_a", "mod_b"), f"mode must be 'mod_a' or 'mod_b', got '{mode}'"

        self.n_aircraft = n_aircraft
        self.mode = mode

        # simulation parameters
        self.D_MAX = 100.0      # corridor length
        self.V_MIN = 1.0        # minimum speed
        self.V_MAX = 5.0        # maximum speed
        self.DELTA_V = 1.0      # speed change per action
        self.D_SEP = 10.0       # minimum pairwise separation
        self.TAU_LAND = 5       # steps pad stays occupied after a landing
        self.MAX_STEPS = 200

        # reward values
        self.R_STEP = -1.0      # small per-step penalty to encourage urgency
        self.R_LAND = 20.0      # bonus when an aircraft successfully lands

        # mod_a: same penalty regardless of why the supervisor intervened
        self.P_GENERIC = -10.0

        # mod_b: penalty depends on violation type so the agent learns what matters more.
        # SEP is harsher than PAD because physical proximity is more dangerous
        # than a simple scheduling/timing error.
        self.P_SEP = -15.0      # separation violation
        self.P_PAD = -8.0       # pad still occupied on arrival

        # using Discrete(27) instead of MultiDiscrete([3,3,3]) for SB3 compatibility
        self.action_space = spaces.Discrete(3 ** self.n_aircraft)

        # observation: [d_i, v_i, eta_i] for each aircraft + tau_pad  →  shape (10,)
        obs_low = np.array(
            [0.0, self.V_MIN, 0.0] * self.n_aircraft + [0.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [self.D_MAX, self.V_MAX, self.D_MAX / self.V_MIN] * self.n_aircraft
            + [float(self.TAU_LAND)],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # state variables – actual values set in reset()
        self.distances = np.zeros(n_aircraft, dtype=np.float32)
        self.speeds = np.zeros(n_aircraft, dtype=np.float32)
        self.tau_pad = 0
        self.active = np.ones(n_aircraft, dtype=bool)   # False once an aircraft has landed
        self.step_count = 0

        # safety supervisor shares the same physical constants as the environment
        self.supervisor = SafetySupervisor(
            d_sep=self.D_SEP,
            v_min=self.V_MIN,
            v_max=self.V_MAX,
            delta_v=self.DELTA_V,
        )

    def _decode_action(self, action: int) -> np.ndarray:
        """
        Convert a flat integer (0–26) to per-aircraft speed commands via base-3 decoding.
        Each base-3 digit d in {0,1,2} maps to command d-1 in {-1, 0, +1}.
        """
        commands = np.zeros(self.n_aircraft, dtype=np.int8)
        for i in range(self.n_aircraft - 1, -1, -1):
            commands[i] = (action % 3) - 1
            action //= 3
        return commands

    def _get_obs(self) -> np.ndarray:
        """Build the flat observation vector from current state."""
        features = []
        for i in range(self.n_aircraft):
            if self.active[i]:
                d = float(self.distances[i])
                v = float(self.speeds[i])
                eta = d / v     # safe: v is always clipped to >= V_MIN = 1.0
            else:
                d, v, eta = 0.0, self.V_MIN, 0.0    # V_MIN keeps obs within declared bounds
            features.extend([d, v, eta])
        features.append(float(self.tau_pad))
        return np.array(features, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset to a new episode with randomised positions and speeds.

        Positions are drawn from staggered windows to preserve initial queue order
        while guaranteeing the separation constraint is never violated at reset:
            Aircraft 0: d in [85, 95]
            Aircraft 1: d in [65, 75]
            Aircraft 2: d in [45, 55]

        Returns obs (shape 10,) and an empty info dict.
        """
        _ = options     # unused, required by Gymnasium API
        super().reset(seed=seed)    # seeds self.np_random

        position_windows = [(85, 95), (65, 75), (45, 55)]
        self.distances = np.array(
            [float(self.np_random.integers(lo, hi + 1)) for lo, hi in position_windows],
            dtype=np.float32,
        )

        # randomise starting speeds so agent has to generalise across conditions
        self.speeds = self.np_random.uniform(2.0, 4.0, size=self.n_aircraft).astype(np.float32)

        self.tau_pad = 0
        self.active = np.ones(self.n_aircraft, dtype=bool)
        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance the simulation by one timestep.

        Proposed commands are validated by the safety supervisor before execution.
        Physics always runs on corrected commands; reward branching then reflects
        whether and why the supervisor had to intervene.

        Returns obs, reward, terminated, truncated, info.
        """
        self.step_count += 1

        commands = self._decode_action(action)

        # supervisor validates proposed commands and returns corrected ones + log
        safe_commands, log = self.supervisor.check(
            self.distances, self.speeds, self.tau_pad, self.active, commands
        )

        # physics always runs on safe (possibly corrected) commands
        for i in range(self.n_aircraft):
            if not self.active[i]:
                continue
            self.speeds[i] = np.clip(
                self.speeds[i] + safe_commands[i] * self.DELTA_V,
                self.V_MIN,
                self.V_MAX,
            )
            self.distances[i] = max(0.0, float(self.distances[i]) - float(self.speeds[i]))

        # decrement pad timer before landing check so the pad can clear
        # and an aircraft can land in the same step
        self.tau_pad = max(0, self.tau_pad - 1)

        # landing detection – only one aircraft can land per step
        landings = 0
        for i in range(self.n_aircraft):
            if self.active[i] and self.distances[i] == 0.0 and self.tau_pad == 0:
                self.active[i] = False
                self.tau_pad = self.TAU_LAND
                landings += 1

        # reward branching: mod_a treats all interventions equally,
        # mod_b penalises SEP harder than PAD since it is physically more dangerous
        reward = self.R_STEP + landings * self.R_LAND
        if log is not None:
            if self.mode == "mod_b":
                reward += self.P_SEP if log.violation_type == "SEP" else self.P_PAD
            else:
                reward += self.P_GENERIC

        terminated = not np.any(self.active)
        truncated = self.step_count >= self.MAX_STEPS

        info = {
            "intervention_occurred": log is not None,
            "intervention_type": log.violation_type if log is not None else "NONE",
            "aircraft_overridden": log.aircraft_idx if log is not None else -1,
            "landings": landings,
        }

        return self._get_obs(), float(reward), terminated, truncated, info
