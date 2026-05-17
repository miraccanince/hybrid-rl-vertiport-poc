"""
Rule-based safety supervisor for the vertiport arrival environment.
Checks proposed joint speed commands for imminent violations and applies
corrective overrides before they reach the physics engine.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class InterventionLog:
    """Carries the result of a supervisor intervention for logging and reward shaping."""
    violation_type: str     # "SEP" or "PAD"
    aircraft_idx: int       # index of the aircraft whose command was overridden


class SafetySupervisor:
    """
    Checks proposed speed commands against two safety rules:
        SEP  –  minimum pairwise separation must stay >= d_sep
        PAD  –  an aircraft cannot land while the pad is still occupied

    Returns corrected commands and an InterventionLog if a violation was found,
    or the original commands and None if everything is safe.
    """

    def __init__(self, d_sep: float, v_min: float, v_max: float, delta_v: float) -> None:
        self.d_sep = d_sep
        self.v_min = v_min
        self.v_max = v_max
        self.delta_v = delta_v

    def _predict_next_distances(
        self,
        distances: np.ndarray,
        speeds: np.ndarray,
        active: np.ndarray,
        commands: np.ndarray,
    ) -> np.ndarray:
        """Simulate one step under the given commands and return predicted distances."""
        next_speeds = np.clip(speeds + commands * self.delta_v, self.v_min, self.v_max)
        # only move active aircraft; landed ones stay at 0
        next_distances = np.where(
            active,
            np.maximum(0.0, distances - next_speeds),
            distances,
        )
        return next_distances

    def check(
        self,
        distances: np.ndarray,
        speeds: np.ndarray,
        tau_pad: int,
        active: np.ndarray,
        commands: np.ndarray,
    ) -> tuple[np.ndarray, InterventionLog | None]:
        """
        Validate proposed commands and return safe commands + optional intervention log.

        PAD is checked before SEP. Only one violation is corrected per call –
        acceptable for a 3-aircraft PoC where simultaneous violations are rare.

        Args:
            distances:  current distances from pad, shape (n,)
            speeds:     current speeds, shape (n,)
            tau_pad:    remaining pad occupation steps
            active:     boolean mask of aircraft still in flight, shape (n,)
            commands:   proposed per-aircraft speed commands in {-1, 0, +1}, shape (n,)

        Returns:
            safe_commands:  corrected command array (copy of input if no violation)
            log:            InterventionLog if an intervention occurred, else None
        """
        safe_commands = commands.copy()
        next_distances = self._predict_next_distances(distances, speeds, active, safe_commands)

        # PAD check: aircraft about to land on an occupied pad
        for i in range(len(distances)):
            if not active[i]:
                continue
            if next_distances[i] == 0.0 and tau_pad > 0:
                # decelerate if possible; if already at v_min, hold (best we can do)
                safe_commands[i] = -1 if speeds[i] > self.v_min else 0
                log = InterventionLog(violation_type="PAD", aircraft_idx=i)
                return safe_commands, log

        # SEP check: any pair of active aircraft getting too close
        n = len(distances)
        for i in range(n):
            for j in range(i + 1, n):
                if not active[i] or not active[j]:
                    continue
                if abs(next_distances[i] - next_distances[j]) < self.d_sep:
                    # trailing aircraft has the larger distance value (farther from pad)
                    trailing = i if distances[i] > distances[j] else j
                    safe_commands[trailing] = -1
                    log = InterventionLog(violation_type="SEP", aircraft_idx=trailing)
                    return safe_commands, log

        return safe_commands, None
