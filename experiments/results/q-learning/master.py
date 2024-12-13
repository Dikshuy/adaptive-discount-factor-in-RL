import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load results from pickle files
def load_results(file_path):
    if not os.path.exists(file_path):
        print(f"Results file not found: {file_path}")
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Function to compute mean and confidence interval
def compute_mean_ci(data):
    mean = np.nanmean(data, axis=0)
    std_err = np.nanstd(data, axis=0) / np.sqrt(data.shape[0])
    ci = 1.96 * std_err
    return mean, ci

# Convergence Analysis: Heatmap
def plot_convergence_heatmap(results_dir, envs, alpha, output_dir):
    data = []
    for env in envs:
        file_path = os.path.join(results_dir, env, f"alpha_{alpha}", "results.pkl")
        results = load_results(file_path)
        if results:
            final_return = np.nanmean(results["exp_ret"], axis=0)[-1]
            data.append(final_return)
        else:
            data.append(np.nan)

    plt.figure(figsize=(8, 6))
    sns.heatmap(np.array(data).reshape(-1, 1), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, xticklabels=[f"alpha={alpha}"], yticklabels=envs)
    plt.title("Convergence Analysis")
    plt.xlabel("Alpha")
    plt.ylabel("Environment")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "convergence_heatmap.png"))
    plt.close()

# Steps Taken Over Time: Error Shade Plot
def plot_steps_over_time(results_dir, envs, alpha, output_dir):
    for env in envs:
        file_path = os.path.join(results_dir, env, f"alpha_{alpha}", "results.pkl")
        results = load_results(file_path)
        if not results:
            continue

        steps_mean, steps_ci = compute_mean_ci(results["steps"])
        plt.figure(figsize=(10, 6))
        plt.plot(steps_mean, label=f"Alpha={alpha}", color="steelblue")
        plt.fill_between(range(len(steps_mean)), steps_mean - steps_ci, steps_mean + steps_ci, color="steelblue", alpha=0.3)
        plt.title(f"Steps Taken Over Time ({env})")
        plt.xlabel("Episodes")
        plt.ylabel("Steps")
        plt.legend()
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{env}_steps_over_time.png"))
        plt.close()

# Initialization Effect Analysis: Bar Plot
def plot_initialization_effect(results_dir, envs, alpha, q_inits, output_dir):
    for env in envs:
        means = []
        cis = []

        for q_init in q_inits:
            file_path = os.path.join(results_dir, env, f"alpha_{alpha}", f"Q_init_{q_init}", "results.pkl")
            results = load_results(file_path)
            if results:
                mean, ci = compute_mean_ci(results["exp_ret"][:, -1])
                means.append(mean)
                cis.append(ci)
            else:
                means.append(np.nan)
                cis.append(np.nan)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=q_inits, y=means, yerr=cis, capsize=5, color="steelblue")
        plt.title(f"Initialization Effect Analysis ({env})")
        plt.xlabel("Q Initialization")
        plt.ylabel("Final Expected Return")
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{env}_initialization_effect.png"))
        plt.close()

# Stability Metrics: Variance Over Time
def plot_stability_metrics(results_dir, envs, alpha, output_dir):
    for env in envs:
        file_path = os.path.join(results_dir, env, f"alpha_{alpha}", "results.pkl")
        results = load_results(file_path)
        if not results:
            continue

        variances = np.nanvar(results["exp_ret"], axis=0)
        plt.figure(figsize=(10, 6))
        plt.plot(variances, label=f"Alpha={alpha}", color="orange")
        plt.title(f"Stability Metrics (Variance) Over Time ({env})")
        plt.xlabel("Episodes")
        plt.ylabel("Variance of Returns")
        plt.legend()
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{env}_stability_metrics.png"))
        plt.close()

# Multi-Metric Comparison: Dual Axis Plot
def plot_multi_metric_comparison(results_dir, envs, alpha, output_dir):
    for env in envs:
        file_path = os.path.join(results_dir, env, f"alpha_{alpha}", "results.pkl")
        results = load_results(file_path)
        if not results:
            continue

        mean_return, ci_return = compute_mean_ci(results["exp_ret"])
        steps_mean, steps_ci = compute_mean_ci(results["steps"])

        fig, ax1 = plt.subplots(figsize=(10, 6))
        color = "tab:blue"
        ax1.set_xlabel("Episodes")
        ax1.set_ylabel("Mean Expected Return", color=color)
        ax1.plot(mean_return, label="Expected Return", color=color)
        ax1.fill_between(range(len(mean_return)), mean_return - ci_return, mean_return + ci_return, color=color, alpha=0.3)
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second y-axis that shares the same x-axis
        color = "tab:green"
        ax2.set_ylabel("Steps", color=color)
        ax2.plot(steps_mean, label="Steps", color=color)
        ax2.fill_between(range(len(steps_mean)), steps_mean - steps_ci, steps_mean + steps_ci, color=color, alpha=0.3)
        ax2.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title(f"Multi-Metric Comparison ({env})")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{env}_multi_metric_comparison.png"))
        plt.close()

# Parameters
environments = ["Empty-10x10-v0", "Empty-Distract-6x6-v0", "Penalty-3x3-v0", "Quicksand-4x4-v0", "Quicksand-Distract-4x4-v0", "TwoRoom-Quicksand-3x5-v0", "Full-4x5-v0", "TwoRoom-Distract-Middle-2x11-v0", "Barrier-5x5-v0"]
q_inits = [0.0, 1.0, 5.0, 10.0]
alpha = 0.5
results_dir = "results2/q_learning/adaptive/Gym-Gridworlds"
output_dir_base = "plots/q_learning_analysis/alpha_0.5"

# Generate Plots
plot_convergence_heatmap(results_dir, environments, alpha, output_dir_base)
plot_steps_over_time(results_dir, environments, alpha, output_dir_base)
plot_initialization_effect(results_dir, environments, alpha, q_inits, output_dir_base)
plot_stability_metrics(results_dir, environments, alpha, output_dir_base)
plot_multi_metric_comparison(results_dir, environments, alpha, output_dir_base)
