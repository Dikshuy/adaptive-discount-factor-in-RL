import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span:])
    return re

def error_shade_plot(ax, data, stepsize, smoothing_window=1, label="", linestyle="-", color=None, linewidth=1.0):
    y = np.nanmean(data, 0)
    x = np.arange(len(y)) * stepsize
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, label=label, linestyle=linestyle, color=color, linewidth=linewidth)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())

def load_results(file_path):
    if not os.path.exists(file_path):
        print(f"Results file not found: {file_path}")
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)

def plot_learning_curves(environments, alphas, save_dir_na, save_dir_a, plot_dir, eval_steps):
    sns.set_style(style="white")
    colorblind_colors = sns.color_palette("colorblind")
    linestyles = ["-", "--", "-.", ":"]

    for env_name in environments:
        env_plot_dir = os.path.join(plot_dir, env_name)
        os.makedirs(env_plot_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"Learning Curves for {env_name} (Q_init=0)", fontsize=14)

        for alpha_idx, alpha in enumerate(alphas):
            results_path = os.path.join(
                save_dir_na, f"{env_name}", f"gamma_0.99", f"Q_init_0.0", f"alpha_{alpha}", "0.99_results.pkl"
            )

            results = load_results(results_path)
            if results is None:
                continue

            exp_ret = results["exp_ret"]
            stepsize = eval_steps

            label = f"alpha={alpha}"
            linestyle = linestyles[alpha_idx % len(linestyles)]
            color = colorblind_colors[alpha_idx % len(colorblind_colors)]

            error_shade_plot(
                ax,
                exp_ret,
                stepsize=stepsize,
                smoothing_window=20,
                label=label,
                linestyle=linestyle,
                color=color,
                linewidth=2.0,
            )

        ax.set_xlabel("Steps")
        ax.set_ylabel("Expected Return")
        ax.legend(loc="best")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.minorticks_on()

        output_path = os.path.join(env_plot_dir, f"{env_name}_learning_curves.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Saved learning curve plot for {env_name}: {output_path}")
        plt.close(fig)


environments = ["Empty-10x10-v0", "Empty-Distract-6x6-v0", "Penalty-3x3-v0", "Quicksand-4x4-v0", "Quicksand-Distract-4x4-v0", "TwoRoom-Quicksand-3x5-v0", "Full-4x5-v0", "TwoRoom-Distract-Middle-2x11-v0", "Barrier-5x5-v0"]
alphas = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
save_dir_na = "results2/q_learning/non_adaptive/Gym-Gridworlds"
save_dir_a = "results2/q_learning/adaptive/Gym-Gridworlds"
plot_dir = "plots/alpha"
eval_steps = 100

plot_learning_curves(environments, alphas, save_dir_na, save_dir_a, plot_dir, eval_steps)
