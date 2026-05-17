"""
Live animation of one vertiport arrival episode.
Aircraft move left-to-right toward the landing pad. The safety supervisor
intervenes in real time – intervention steps are highlighted in red.

Run from project root:
    python visualize.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from env.vertiport_env import VertiportEnv

COLORS = ["tab:blue", "tab:orange", "tab:green"]
DELAY  = 0.18   # seconds per frame (lower = faster)
MODE   = "mod_b"
SEED   = 3


def draw_frame(ax, env, step: int, info: dict, reward: float) -> None:
    ax.clear()

    # corridor background
    ax.fill_betweenx([-0.6, 0.6], 0, env.D_MAX, color="#f0f0f0")
    ax.axhline(0, color="#cccccc", linewidth=1)

    # separation danger zones between active aircraft pairs
    active_idx = [i for i in range(env.n_aircraft) if env.active[i]]
    for i in range(len(active_idx)):
        for j in range(i + 1, len(active_idx)):
            a, b = active_idx[i], active_idx[j]
            gap = abs(env.distances[a] - env.distances[b])
            if gap < env.D_SEP:
                lo = min(env.distances[a], env.distances[b])
                hi = max(env.distances[a], env.distances[b])
                ax.fill_betweenx([-0.5, 0.5], lo, hi, color="red", alpha=0.15)

    # landing pad – shown as green (free) or red (occupied)
    pad_color = "#e74c3c" if env.tau_pad > 0 else "#2ecc71"
    pad = mpatches.FancyBboxPatch(
        (env.D_MAX - 1.5, -0.55), 3, 1.1,
        boxstyle="round,pad=0.1", color=pad_color, zorder=3
    )
    ax.add_patch(pad)
    pad_label = f"PAD\nτ={env.tau_pad}" if env.tau_pad > 0 else "PAD\nfree"
    ax.text(env.D_MAX, 0, pad_label, ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=4)

    # aircraft markers
    for i in range(env.n_aircraft):
        if env.active[i]:
            # x position: aircraft move from left (d=D_MAX) toward right (d=0=pad)
            x = env.D_MAX - env.distances[i]
            ax.plot(x, 0, "o", color=COLORS[i], markersize=22, zorder=5)
            ax.text(x, 0, f"AC{i}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=6)
            ax.text(x, 0.72,
                    f"d={env.distances[i]:.0f}  v={env.speeds[i]:.1f}",
                    ha="center", va="bottom", fontsize=8, color=COLORS[i])
        else:
            # landed – show faded marker near pad
            ax.plot(env.D_MAX + 0.5, 0, "o", color=COLORS[i],
                    markersize=14, alpha=0.25, zorder=5)
            ax.text(env.D_MAX + 0.5, -0.72, f"AC{i}\nlanded",
                    ha="center", va="top", fontsize=7, color=COLORS[i], alpha=0.5)

    # axis formatting
    ax.set_xlim(-3, env.D_MAX + 6)
    ax.set_ylim(-1.4, 1.6)
    ax.set_yticks([])
    ax.set_xlabel("Position along corridor  (pad = right end)", fontsize=9)

    # title – red on intervention, normal otherwise
    if info.get("intervention_occurred", False):
        itype = info["intervention_type"]
        ac    = info["aircraft_overridden"]
        title = f"Step {step}  |  SUPERVISOR INTERVENED  –  {itype}  on  AC{ac}  |  reward {reward:+.0f}"
        ax.set_title(title, color="#c0392b", fontsize=10, fontweight="bold")
    else:
        title = f"Step {step}  |  no intervention  |  reward {reward:+.0f}"
        ax.set_title(title, color="#2c3e50", fontsize=10)

    # legend
    patches = [mpatches.Patch(color=COLORS[i], label=f"AC{i}") for i in range(env.n_aircraft)]
    patches += [
        mpatches.Patch(color="#2ecc71", label="pad free"),
        mpatches.Patch(color="#e74c3c", label="pad occupied"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.8)


def run() -> None:
    env = VertiportEnv(mode=MODE)
    obs, _ = env.reset(seed=SEED)

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.suptitle(f"Vertiport Arrival Simulation  –  {MODE.upper()}  (random policy + safety supervisor)",
                 fontsize=11)
    plt.ion()

    step = 0
    total_reward = 0.0
    interventions = 0
    empty_info = {"intervention_occurred": False}

    draw_frame(ax, env, step, empty_info, 0.0)
    plt.tight_layout()
    plt.pause(DELAY * 3)

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        total_reward += reward
        if info["intervention_occurred"]:
            interventions += 1

        draw_frame(ax, env, step, info, reward)
        plt.tight_layout()

        # pause longer on intervention steps so they are visible
        pause = DELAY * 2.5 if info["intervention_occurred"] else DELAY
        plt.pause(pause)

        if terminated or truncated:
            break

    # final frame
    ax.set_title(
        f"Episode complete  –  {step} steps  |  "
        f"total reward {total_reward:+.0f}  |  interventions {interventions}",
        fontsize=10, color="#27ae60"
    )
    plt.tight_layout()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run()
