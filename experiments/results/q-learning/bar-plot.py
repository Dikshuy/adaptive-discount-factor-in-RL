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

def bar_plot_comparison(environments, q_inits, alphas, save_dir_na, save_dir_a, base_plot_dir):
    sns.set_style(style="white")
    adaptive_label = "adaptive_gamma"
    fixed_label = "gamma=0.99"
    fixed_label2 = "gamma=0.5"

    for alpha in alphas:
        plot_dir = os.path.join(base_plot_dir, f"alpha_{alpha}")
        os.makedirs(plot_dir, exist_ok=True)

        for env_name in environments:
            adaptive_means = []
            adaptive_cis = []
            fixed_means = []
            fixed_cis = []
            fixed_means2 = []
            fixed_cis2 = []
            labels = []

            for init in q_inits:
                adaptive_path = os.path.join(
                    save_dir_a, f"{env_name}", f"gamma_0.5", f"Q_init_{init}", f"alpha_{alpha}", "0.5_results.pkl"
                )
                fixed_path = os.path.join(
                    save_dir_na, f"{env_name}", f"gamma_0.99", f"Q_init_{init}", f"alpha_{alpha}", "0.99_results.pkl"
                )
                fixed_path2 = os.path.join(
                    save_dir_na, f"{env_name}", f"gamma_0.5", f"Q_init_{init}", f"alpha_{alpha}", "0.5_results.pkl"
                )

                adaptive_results = load_results(adaptive_path)
                fixed_results = load_results(fixed_path)
                fixed_results2 = load_results(fixed_path2)

                adaptive_mean, adaptive_ci = extract_performance(adaptive_results)
                fixed_mean, fixed_ci = extract_performance(fixed_results)
                fixed_mean2, fixed_ci2 = extract_performance(fixed_results2)

                adaptive_means.append(adaptive_mean)
                adaptive_cis.append(adaptive_ci)
                fixed_means.append(fixed_mean)
                fixed_cis.append(fixed_ci)
                fixed_means2.append(fixed_mean2)
                fixed_cis2.append(fixed_ci2)
                labels.append(f"Q_init={init}")

            x = np.arange(len(q_inits))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 6))
            rects1 = ax.bar(x - width / 3, adaptive_means, width, label=adaptive_label, color="steelblue", yerr=adaptive_cis, capsize=5)
            rects2 = ax.bar(x + width, fixed_means, width, label=fixed_label, color="orange", yerr=fixed_cis, capsize=5)
            rects3 = ax.bar(x + width / 3, fixed_means2, width, label=fixed_label2, color="green", yerr=fixed_cis2, capsize=5)

            ax.set_xlabel("Q-Value Initialization")
            ax.set_ylabel("Mean Expected Return")
            ax.set_title(f"Performance Comparison for {env_name}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            def annotate_bars(rects):
                for rect in rects:
                    height = rect.get_height()
                    if not np.isnan(height):
                        ax.annotate(
                            f"{height:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                        )

            annotate_bars(rects1)
            annotate_bars(rects2)
            annotate_bars(rects3)


            output_path = os.path.join(plot_dir, f"{env_name}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            print(f"Saved comparison plot for {env_name}: {output_path}")
            plt.close(fig)

environments = ["Empty-10x10-v0", "Empty-Distract-6x6-v0", "Penalty-3x3-v0", "Quicksand-4x4-v0", "Quicksand-Distract-4x4-v0", "TwoRoom-Quicksand-3x5-v0", "Full-4x5-v0", "TwoRoom-Distract-Middle-2x11-v0", "Barrier-5x5-v0"]
q_inits = [0.0, 5.0, 10.0]
alphas = [0.5]
save_dir_na = "results2/q_learning/non_adaptive/Gym-Gridworlds"
save_dir_a = "results2/q_learning/adaptive_0.5/Gym-Gridworlds"
base_plot_dir = "plots/bar_plots"

bar_plot_comparison(environments, q_inits, alphas, save_dir_na, save_dir_a, base_plot_dir)