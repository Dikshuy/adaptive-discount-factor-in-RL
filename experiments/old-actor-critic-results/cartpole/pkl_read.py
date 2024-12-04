import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

# gammas and q inits
gammas = [0.1,0.5,0.8,0.95,0.99]
q_initializations = [100.0, 20.0, 10.0, 1.0, -5.0]

base_path = 'experiments/old_actor_critic_results/cartpole/cartpole_init_'

def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
    return re

def error_shade_plot(ax, data, stepsize, smoothing_window=1, label="", linestyle="-", color=None):
    y = np.nanmean(data, 0)
    x = np.arange(len(y)) * stepsize
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, label=label, linestyle=linestyle, color=color)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])  # CI
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())

# def plot_with_confidence(item, ax, color, label, marker):
#     mean_item = np.mean(item, axis=0)
    
#     lower_bound = np.percentile(item, 2.5, axis=0)
#     upper_bound = np.percentile(item, 97.5, axis=0)
    
#     episodes = np.arange(item.shape[1])

#     ax.plot(episodes, mean_item, color=color, label=label, zorder=2)
#     ax.fill_between(episodes, lower_bound, upper_bound, color=color, alpha=0.1, zorder=1)
    
#     return ax

colorblind_colors = sns.color_palette("colorblind")

for q_init_idx, q_init in enumerate(q_initializations):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_prop_cycle(cycler('color', colorblind_colors))

    for gamma_idx, gamma in enumerate(gammas):
        file_path = f'{base_path}{q_init}_gamma_{gamma}_results.pkl'
        try:
            with open(file_path, 'rb') as file:
                results = pickle.load(file)

            eval_returns = np.array(results['evaluation returns'])
            
            error_shade_plot(
                ax,
                eval_returns,
                stepsize=1,
                smoothing_window=20,
                label=f"Gamma = {gamma}",
                linestyle="-",
                color=colorblind_colors[gamma_idx]
            )
            # ax = plot_with_confidence(
            #     eval_returns,
            #     ax=ax,
            #     color=colorblind_colors[gamma_idx],
            #     label=f"Gamma = {gamma}",
            #     marker=None
            # )
        
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

    ax.set_xlabel('Episodes')
    ax.set_ylabel('Returns')
    ax.set_title(f'Evaluation Returns (Q_init = {q_init})')
    ax.legend(loc='best')
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()