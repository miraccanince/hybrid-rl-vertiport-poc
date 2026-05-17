"""
Training and benchmarking script for the hybrid vertiport RL PoC.

Trains PPO on both mod_a and mod_b environments and generates a comparison
plot showing learning curve convergence and cumulative safety interventions.

Run from project root:
    python training/train_benchmark.py
"""

import sys
import os

# ensure project root is on the path when running this script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env.vertiport_env import VertiportEnv

TOTAL_TIMESTEPS = 50_000
SEED = 42


class InterventionCallback(BaseCallback):
    """
    Hooks into the SB3 training loop to track safety interventions and episode
    rewards from our custom info dict at every timestep.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

        # cumulative intervention counts, one entry per timestep
        self.sep_counts: list[int] = []
        self.pad_counts: list[int] = []

        # episode-level tracking for the learning curve
        self.episode_rewards: list[float] = []
        self.episode_end_steps: list[int] = []

        self._sep = 0
        self._pad = 0
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        # SB3 wraps envs in a VecEnv, so infos is a list – grab index 0
        info = self.locals["infos"][0]

        itype = info.get("intervention_type", "NONE")
        if itype == "SEP":
            self._sep += 1
        elif itype == "PAD":
            self._pad += 1

        self.sep_counts.append(self._sep)
        self.pad_counts.append(self._pad)

        self._ep_reward += float(self.locals["rewards"][0])

        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_end_steps.append(self.num_timesteps)
            self._ep_reward = 0.0

        return True


def smooth(values: list[float], window: int = 15) -> np.ndarray:
    """Moving average to reduce noise on episode reward curves."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def train(mode: str) -> tuple[InterventionCallback, PPO]:
    """Train PPO on the given supervision mode and return callback + trained model."""
    env = VertiportEnv(mode=mode)
    model = PPO("MlpPolicy", env, seed=SEED, verbose=0)
    callback = InterventionCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    env.close()
    return callback, model


def plot_results(cb_a: InterventionCallback, cb_b: InterventionCallback) -> None:
    """Generate and save the mod_a vs mod_b comparison figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Mod A vs Mod B – Hybrid RL Safety Supervisor PoC\n"
        "PPO, 3-aircraft vertiport, 50 000 timesteps",
        fontsize=12,
    )

    # ── Left: learning curves ──
    r_a = smooth(cb_a.episode_rewards)
    r_b = smooth(cb_b.episode_rewards)
    # episode_end_steps are trimmed by the smoothing window, so align x-axis
    x_a = cb_a.episode_end_steps[len(cb_a.episode_end_steps) - len(r_a):]
    x_b = cb_b.episode_end_steps[len(cb_b.episode_end_steps) - len(r_b):]

    ax1.plot(x_a, r_a, label="Mod A (generic penalty)", color="tab:blue")
    ax1.plot(x_b, r_b, label="Mod B (reason-coded penalty)", color="tab:orange")
    ax1.set_title("Learning Curve (smoothed episode reward)")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Episode Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Right: cumulative interventions ──
    # SB3 collects full rollouts so actual steps may slightly exceed TOTAL_TIMESTEPS
    steps_a = np.arange(1, len(cb_a.sep_counts) + 1)
    steps_b = np.arange(1, len(cb_b.sep_counts) + 1)
    ax2.plot(steps_a, cb_a.sep_counts, label="Mod A – SEP", color="tab:blue",   linestyle="-")
    ax2.plot(steps_a, cb_a.pad_counts, label="Mod A – PAD", color="tab:blue",   linestyle="--")
    ax2.plot(steps_b, cb_b.sep_counts, label="Mod B – SEP", color="tab:orange", linestyle="-")
    ax2.plot(steps_b, cb_b.pad_counts, label="Mod B – PAD", color="tab:orange", linestyle="--")
    ax2.set_title("Cumulative Safety Interventions")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Cumulative Count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "results/comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    print(f"Training Mod A  ({TOTAL_TIMESTEPS:,} timesteps)...")
    cb_a, _ = train("mod_a")
    print(f"  done – episodes: {len(cb_a.episode_rewards)}  SEP: {cb_a._sep}  PAD: {cb_a._pad}")

    print(f"\nTraining Mod B  ({TOTAL_TIMESTEPS:,} timesteps)...")
    cb_b, _ = train("mod_b")
    print(f"  done – episodes: {len(cb_b.episode_rewards)}  SEP: {cb_b._sep}  PAD: {cb_b._pad}")

    print("\nGenerating comparison plot...")
    plot_results(cb_a, cb_b)
