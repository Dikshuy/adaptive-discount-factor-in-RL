import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(file_path):
    if not os.path.exists(file_path):
        print(f"Results file not found: {file_path}")
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)

def extract_performance(results):
    if results is None:
        return np.nan, np.nan
    mean = np.nanmean(results["exp_ret"])
    std_err = np.nanstd(results["exp_ret"]) / np.sqrt(len(results["exp_ret"]))
    conf_interval = 1.96 * std_err
    return mean, conf_interval

def line_plot_comparison(environments, q_inits, alphas, save_dir_na, save_dir_a, output_file):
    sns.set_style(style="white")
    adaptive_label = "adaptive_gamma"
    fixed_label = "gamma=0.99"
    # fixed_label2 = "gamma=0.5"

    for alpha in alphas:
        # Prepare data for line plot
        adaptive_means = {q_init: [] for q_init in q_inits}
        adaptive_cis = {q_init: [] for q_init in q_inits}
        fixed_means = {q_init: [] for q_init in q_inits}
        fixed_cis = {q_init: [] for q_init in q_inits}
        # fixed_means2 = {q_init: [] for q_init in q_inits}
        # fixed_cis2 = {q_init: [] for q_init in q_inits}

        for env_name in environments:
            for q_init in q_inits:
                adaptive_path = os.path.join(
                    save_dir_a, f"{env_name}", f"gamma_0.5", f"Q_init_{q_init}", f"alpha_{alpha}", "0.5_results.pkl"
                )
                fixed_path = os.path.join(
                    save_dir_na, f"{env_name}", f"gamma_0.99", f"Q_init_{q_init}", f"alpha_{alpha}", "0.99_results.pkl"
                )
                # fixed_path2 = os.path.join(
                #     save_dir_na, f"{env_name}", f"gamma_0.5", f"Q_init_{q_init}", f"alpha_{alpha}", "0.5_results.pkl"
                # )

                adaptive_results = load_results(adaptive_path)
                fixed_results = load_results(fixed_path)
                # fixed_results2 = load_results(fixed_path2)

                adaptive_mean, adaptive_ci = extract_performance(adaptive_results)
                fixed_mean, fixed_ci = extract_performance(fixed_results)
                # fixed_mean2, fixed_ci2 = extract_performance(fixed_results2)

                adaptive_means[q_init].append(adaptive_mean)
                adaptive_cis[q_init].append(adaptive_ci)
                fixed_means[q_init].append(fixed_mean)
                fixed_cis[q_init].append(fixed_ci)
                # fixed_means2[q_init].append(fixed_mean2)
                # fixed_cis2[q_init].append(fixed_ci2)

        # Plotting
        x = np.arange(len(environments))  # X-axis positions

        plt.figure(figsize=(12, 8))
        for q_init in q_inits:
            # Adaptive Gamma
            plt.errorbar(x, adaptive_means[q_init], label=f"Q_init={q_init} ({adaptive_label})", color="red",
                         marker='o', linestyle='-', capsize=4)
            # Fixed Gamma 0.99
            plt.errorbar(x, fixed_means[q_init], label=f"Q_init={q_init} ({fixed_label})", color="blue",
                         marker='^', linestyle='--', capsize=4)
            # Fixed Gamma 0.5
            # plt.errorbar(x, fixed_means2[q_init], label=f"Q_init={q_init} ({fixed_label2})", color="green",
            #              marker='s', linestyle='-.', capsize=4)

        # Add labels, title, and legend
        plt.xticks(x, environments, rotation=45, ha="right")
        plt.xlabel("Environments")
        plt.ylabel("Mean Expected Return")
        plt.title(f"Performance Comparison for Alpha={alpha}")
        plt.legend()
        plt.tight_layout()

        # Save and show plot
        plt.savefig(output_file, dpi=300)
        print(f"Saved line plot comparison: {output_file}")
        # plt.show()

# Parameters
environments = ["Empty-10x10-v0", "Empty-Distract-6x6-v0", "Penalty-3x3-v0", "Quicksand-4x4-v0", "Quicksand-Distract-4x4-v0", "TwoRoom-Quicksand-3x5-v0", "Full-4x5-v0", "TwoRoom-Distract-Middle-2x11-v0", "Barrier-5x5-v0"]
q_inits = [0.0, 10.0]
alphas = [0.5]
save_dir_na = "results2/q_learning/non_adaptive/Gym-Gridworlds"
save_dir_a = "results2/q_learning/adaptive_0.5/Gym-Gridworlds"
output_file = "plots/line_plot_comparison.png"

# Generate line plot
line_plot_comparison(environments, q_inits, alphas, save_dir_na, save_dir_a, output_file)
