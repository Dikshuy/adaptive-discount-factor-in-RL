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
        re[-i] = np.average(arr[-i - span :])
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

def plot_results(environment, gammas, q_initializations, base_path, save_dir):
    colorblind_colors = sns.color_palette("colorblind")
    os.makedirs(save_dir, exist_ok=True)
    plt.rc('axes', prop_cycle=cycler('color', colorblind_colors))
    
    for _, q_init in enumerate(q_initializations):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_prop_cycle(cycler('color', colorblind_colors))

        # adaptive gamma
        file_path = f'{base_path}{environment}_init_{q_init}_adaptive_gamma_results.pkl'
        try:
            with open(file_path, 'rb') as file:
                results = pickle.load(file)
            eval_returns = np.array(results['evaluation returns'])
            error_shade_plot(
                ax,
                eval_returns,
                stepsize=500,
                smoothing_window=1,
                label="γ = adaptive γ",
                linestyle="-",
                color=colorblind_colors[-1],
                linewidth=3.0
            )
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

        # non-adaptive gamma
        for gamma_idx, gamma in enumerate(gammas):
            file_path = f'{base_path}{environment}_init_{q_init}_gamma_{gamma}_results.pkl'
            try:
                with open(file_path, 'rb') as file:
                    results = pickle.load(file)
                eval_returns = np.array(results['evaluation returns'])
                error_shade_plot(
                    ax,
                    eval_returns,
                    stepsize=500,
                    smoothing_window=1,
                    label=f"γ = {gamma}",
                    linestyle="-",
                    color=colorblind_colors[gamma_idx],
                    linewidth=1.0
                )
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Eval Returns')
        ax.set_title(f'LFA AC - {environment.capitalize()}')
        ax.legend(fontsize="x-small", loc="best")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.minorticks_on()
        plt.tight_layout()
    
        save_path = os.path.join(save_dir, f"{environment}_init_{q_init}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot: {save_path}")
        plt.close()

# Parameters
environments = ['cartpole']#, 'pendulum', 'mountain_car']
gammas = [0.1, 0.25, 0.5, 0.75, 0.95, 0.99]
q_initializations = [5.0]#[-1, 0, 1]
base_path = 'data/'
save_dir = 'plots/'

# Generate plots for all environments
for env in environments:
    plot_results(env, gammas, q_initializations, base_path, os.path.join(save_dir, env))
