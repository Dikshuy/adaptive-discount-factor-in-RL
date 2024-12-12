import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

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

def plot_separate_gamma_results(environments, gamma_values, plot_dir, eval_steps):
    colorblind_colors = sns.color_palette("colorblind")
    linestyles = ["-"]
    plt.rc("axes", prop_cycle=cycler("color", colorblind_colors))
    colors = sns.color_palette("colorblind", len(gamma_values))

    for _, env_name in enumerate(environments):
        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f"{env_name}", fontsize=14)

        env_dir_na = os.path.join(save_dir_na, env_name)
        env_dir_a = os.path.join(save_dir_a, env_name)
        env_plot_dir = os.path.join(plot_dir, env_name)

        os.makedirs(env_plot_dir, exist_ok=True)

        gamma_results_path = os.path.join(env_dir_a, f"gamma_0.1", f"Q_init_{Q_init}", f"0.1_results.pkl")

        if not os.path.exists(gamma_results_path):
            print(f"Results file not found: {gamma_results_path}")
            continue

        with open(gamma_results_path, "rb") as f:
            results = pickle.load(f)

        exp_ret = results["exp_ret"]
        exp_steps = results["steps"]
        stepsize = eval_steps

        label = f"γ=adaptive γ"

        error_shade_plot(
            axs[0],
            exp_ret,
            stepsize=stepsize,
            smoothing_window=20,
            label=label,
            linestyle="--",
            color=colorblind_colors[-1],
            linewidth=3.0,
        )

        error_shade_plot(
            axs[1],
            exp_steps,
            stepsize=stepsize,
            smoothing_window=20,
            label=label,
            linestyle="--",
            color=colorblind_colors[-1],
            linewidth=3.0,
        )

        for gamma_idx, gamma in enumerate(gamma_values):
            gamma_results_path = os.path.join(env_dir_na, f"gamma_{gamma}", f"Q_init_{Q_init}", f"{gamma}_results.pkl")

            if not os.path.exists(gamma_results_path):
                print(f"Results file not found: {gamma_results_path}")
                continue

            with open(gamma_results_path, "rb") as f:
                results = pickle.load(f)

            exp_ret = results["exp_ret"]
            exp_steps = results["steps"]
            stepsize = eval_steps

            label = f"γ={gamma}"
            linestyle = linestyles[gamma_idx % len(linestyles)]
            color = colors[gamma_idx]

            error_shade_plot(
                axs[0],
                exp_ret,
                stepsize=stepsize,
                smoothing_window=20,
                label=label,
                linestyle=linestyle,
                color=color,
                linewidth=1.0,
            )

            error_shade_plot(
                axs[1],
                exp_steps,
                stepsize=stepsize,
                smoothing_window=20,
                label=label,
                linestyle=linestyle,
                color=color,
                linewidth=1.0,
            )

        axs[0].set_xlabel("Steps")
        axs[0].set_ylabel("Expected Return")
        axs[0].legend(loc="best")
        axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)
        axs[0].minorticks_on()
        axs[0].set_ylim([-5, 1.4])

        axs[1].set_xlabel("Steps")
        axs[1].set_ylabel("Steps Taken")
        axs[1].legend(loc="best")
        axs[1].grid(True, which="both", linestyle="--", linewidth=0.5)
        axs[1].minorticks_on()

        output_path = os.path.join(env_plot_dir, f"{env_name}_{Q_init}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Saved plot for {env_name}: {output_path}")
        plt.close(fig)

environments = ["EASY_SPARSE", "EASY_MEDIUM", "EASY_DENSE", "MODERATE_SPARSE", "MODERATE_MEDIUM", "MODERATE_DENSE", "DIFFICULT_SPARSE", "DIFFICULT_MEDIUM", "DIFFICULT_DENSE"]
gammas = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
save_dir_na = "q_learning/non_adaptive"
save_dir_a = "q_learning/adaptive"
plot_dir = "plots/q_learning"
Q_init = 0.0
eval_steps = 100

plot_separate_gamma_results(environments, gammas, plot_dir, eval_steps)
