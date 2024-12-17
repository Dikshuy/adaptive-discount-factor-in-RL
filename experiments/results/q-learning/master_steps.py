import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import collections
import seaborn as sns

def load_results(file_path):
    if not os.path.exists(file_path):
        print(f"Results file not found: {file_path}")
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)

def extract_performance(env, results):
    if results is None:
        return np.nan, np.nan
    steps = results["steps"]

    if env == "Empty-10x10-v0":
        optimal = []
        seed = 0
        while seed < 50:
            count = collections.Counter(steps[seed])
            optimal.append(count[19] / 600 * 100)
            seed += 1
        
    if env == "Empty-Distract-6x6-v0":
        optimal = []
        seed = 0
        while seed < 50:
            count = collections.Counter(steps[seed])
            optimal.append(count[11] / 600 * 100)
            seed += 1

    if env == "Penalty-3x3-v0":
        optimal = []
        seed = 0
        while seed < 50:
            count = collections.Counter(steps[seed])
            optimal.append(count[7] / 600 * 100)
            seed += 1

    if env == "Quicksand-4x4-v0":
        optimal = []
        seed = 0
        while seed < 50:
            count = collections.Counter(steps[seed])
            optimal.append(count[10] / 600 * 100)
            seed += 1

    if env == "Quicksand-Distract-4x4-v0":
        optimal = []
        seed = 0
        while seed < 50:
            count = collections.Counter(steps[seed])
            optimal.append(count[10] / 600 * 100)
            seed += 1

    if env == "TwoRoom-Quicksand-3x5-v0":
        optimal = []
        seed = 0
        while seed < 50:
            count = collections.Counter(steps[seed])
            optimal.append(count[9] / 600 * 100)
            seed += 1

    if env == "Full-4x5-v0":
        optimal = []
        seed = 0
        while seed < 50:
            count = collections.Counter(steps[seed])
            optimal.append(count[11] / 600 * 100)
            seed += 1

    if env == "TwoRoom-Distract-Middle-2x11-v0":
        optimal = []
        seed = 0
        while seed < 50:
            count = collections.Counter(steps[seed])
            optimal.append(count[7] / 600 * 100)
            seed += 1

    if env == "Barrier-5x5-v0":
        optimal = []
        seed = 0
        while seed < 50:
            count = collections.Counter(steps[seed])
            optimal.append(count[9] / 600 * 100)
            seed += 1

    mean = np.nanmean(optimal)
    return mean


def vertical_bar_plot(environments, save_dir_na, save_dir_a, base_plot_dir):
    sns.set_style(style="white")
    adaptive_label = "Adaptive γ"
    fixed_label = "Fixed γ = 0.99"
    q_init = 10.0
    alpha = 0.5

    plot_dir = os.path.join(base_plot_dir, "vertical_bar_plot")
    os.makedirs(plot_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 14})

    adaptive_means = []
    fixed_means = []
    labels = []

    for env_name in environments:
        adaptive_path = os.path.join(
            save_dir_a, f"{env_name}", "gamma_0.5", f"Q_init_{q_init}", f"alpha_{alpha}", "0.5_results.pkl"
        )
        fixed_path = os.path.join(
            save_dir_na, f"{env_name}", "gamma_0.99", f"Q_init_{q_init}", f"alpha_{alpha}", "0.99_results.pkl"
        )

        adaptive_results = load_results(adaptive_path)
        fixed_results = load_results(fixed_path)

        adaptive_mean = extract_performance(env_name, adaptive_results)
        fixed_mean = extract_performance(env_name, fixed_results)

        adaptive_means.append(adaptive_mean)
        fixed_means.append(fixed_mean)
        labels.append(env_name.replace("-v0", "").replace("-", "\n"))

    x = np.arange(len(environments)) 
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width / 2, adaptive_means, width, color="steelblue", label=adaptive_label, alpha=0.85)
    rects2 = ax.bar(x + width / 2, fixed_means, width, color="orange", label=fixed_label, alpha=0.85)

    ax.set_ylabel("Percentage of Optimal Trajectories (%)")
    # ax.set_xlabel("Environments")
    # ax.set_title("Performance Comparison Across Environments", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontfamily="monospace", fontsize=11)
    ax.legend(prop={'size': 14}, loc="upper left")

    plt.tight_layout()
    output_path = os.path.join(plot_dir, f"{q_init}_steps.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved vertical bar plot: {output_path}")


environments = [
    "Empty-10x10-v0",
    "Empty-Distract-6x6-v0",
    "Penalty-3x3-v0",
    "Quicksand-Distract-4x4-v0",
    "TwoRoom-Quicksand-3x5-v0",
    "Full-4x5-v0",
    "TwoRoom-Distract-Middle-2x11-v0",
    "Barrier-5x5-v0"
]

save_dir_na = "results2/q_learning/non_adaptive/Gym-Gridworlds"
save_dir_a = "results2/q_learning/adaptive_0.5/Gym-Gridworlds"
base_plot_dir = "plots/bar_plots"

vertical_bar_plot(
    environments=environments,
    save_dir_na=save_dir_na,
    save_dir_a=save_dir_a,
    base_plot_dir=base_plot_dir,
)