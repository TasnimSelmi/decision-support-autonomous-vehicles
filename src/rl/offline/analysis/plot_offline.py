import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

log_path = Path("outputs/rl/logs/dqn_training_log.csv")
plot_dir = Path("outputs/rl/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(log_path)

# =========================
# 1. Reward per scenario
# =========================
plt.figure()

for scenario in df["scenario_name"].unique():
    sub = df[df["scenario_name"] == scenario]
    plt.plot(sub["episode"], sub["total_reward"], label=scenario)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Scenario")
plt.legend()
plt.grid()

plt.savefig(plot_dir / "reward_per_scenario.png")
plt.close()

# =========================
# 2. Smoothed reward per scenario
# =========================
plt.figure()

for scenario in df["scenario_name"].unique():
    sub = df[df["scenario_name"] == scenario].copy()
    sub["smoothed"] = sub["total_reward"].rolling(window=10).mean()
    plt.plot(sub["episode"], sub["smoothed"], label=scenario)

plt.xlabel("Episode")
plt.ylabel("Smoothed Reward")
plt.title("Smoothed Reward per Scenario")
plt.legend()
plt.grid()

plt.savefig(plot_dir / "smoothed_reward_per_scenario.png")
plt.close()

# =========================
# 3. Boxplot (VERY IMPORTANT)
# =========================
plt.figure()

df.boxplot(column="total_reward", by="scenario_name")

plt.title("Reward Distribution per Scenario")
plt.suptitle("")  # remove default title
plt.xlabel("Scenario")
plt.ylabel("Total Reward")

plt.savefig(plot_dir / "reward_boxplot.png")
plt.close()

print("Scenario plots saved to:", plot_dir)