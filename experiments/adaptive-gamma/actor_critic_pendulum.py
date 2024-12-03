import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns
from tqdm import tqdm
import argparse
import itertools
import pickle
import os

np.set_printoptions(precision=3, suppress=True)

def cantor_pairing(x, y):
    return int(0.5 * (x + y) * (x + y + 1) + y)

def rbf_features(x: np.array, c: np.array, s: np.array) -> np.array:
    return np.exp(-(((x[:, None] - c[None]) / s[None])**2).sum(-1) / 2.0)

def expected_return(env, weights, sigma, episodes=10):
    G = np.zeros(episodes)
    T = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            phi = get_phi(s)
            a = gaussian_action(phi, weights, sigma)
            a_clip = np.clip(a, env.action_space.low, env.action_space.high)
            s_next, r, terminated, truncated, _ = env.step(a_clip)
            done = terminated or truncated
            G[e] += r
            s = s_next
            t += 1
            if done:
                T[e] = t
    return G.mean(), T.mean()

def gaussian_action(phi, weights, sigma: np.array):
    mu = np.dot(phi, weights)
    return np.random.normal(mu, sigma**2)

def dlog_gaussian_probs(phi, weights, sigma, action: np.array):
    mu = np.dot(phi, weights)
    return phi * (action - mu) / (sigma**2)

def actor_critic(gamma, init, seed, alpha_actor, alpha_critic, episodes_eval, eval_steps, max_steps, adaptive_gamma):
    actor_weights = np.zeros((phi_dummy.shape[1], action_dim))
    critic_weights = np.zeros(phi_dummy.shape[1]) + init

    train_rets, train_lens = [], []
    eval_rets, eval_lens = [], []
    t_len, t_ret = 0, 0

    sigma = 1.0  # for Gaussian
    tot_steps = 0
    episodes = 0
    exp_return_history = np.zeros(max_steps)
    td_error_history = np.zeros(max_steps)
    exp_return, exp_len= expected_return(env_eval, actor_weights, 0, episodes_eval)
    eval_rets.append(exp_return)
    eval_lens.append(exp_len)
    pbar = tqdm(total=max_steps)

    while tot_steps < max_steps:
        s, _ = env.reset(seed = seed)
        done = False
        T = 0

        while not done and tot_steps < max_steps:
            phi = get_phi(s)
            a = gaussian_action(phi, actor_weights, sigma)
            a_clip = np.clip(a, env.action_space.low, env.action_space.high)
            s_next, r, terminated, truncated, _ = env.step(a_clip)
            done = terminated or truncated
            phi_next = get_phi(s_next)
           
            v = np.dot(phi, critic_weights)
            v_next = np.dot(phi_next, critic_weights)
            
            td_error = r + gamma * v_next * (1 - terminated) - v
            critic_weights += alpha_critic * td_error * phi.flatten()
            dlog_pi = dlog_gaussian_probs(phi, actor_weights, sigma, a)
            actor_weights += alpha_actor * td_error * dlog_pi.reshape(-1, 1)

            s = s_next
            T += 1
            tot_steps += 1

            exp_return_history[tot_steps-1] = exp_return
            td_error_history[tot_steps-1] = abs(td_error)

            if done:
                train_rets.append(t_ret)
                train_lens.append(t_len)
                t_len, t_ret = 0, 0
                episodes += 1
                if adaptive_gamma:
                    gamma = min(gamma + episodes * gamma / max_steps, 0.95) 

            if tot_steps % eval_steps == 0:
                exp_return, eval_len = expected_return(env_eval, actor_weights, 0, episodes_eval)
                eval_rets.append(exp_return)
                eval_lens.append(eval_len)   
            
            sigma = max(sigma - 1 / max_steps, 0.01)

            pbar.set_description(
                f"G: {exp_return:.3f}"
        )
        pbar.update(T)

    pbar.close()
    return exp_return_history, (train_rets, train_lens), (eval_rets, eval_lens), td_error_history

def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
    return re

def error_shade_plot(ax, data, stepsize, smoothing_window=1, label="", linestyle="-", **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, label=label, linestyle=linestyle, **kwargs)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Actor-Critic for Pendulum-v1")
    parser.add_argument("--gamma_values", type=float, nargs="+", default=[0.1, 0.5, 0.8, 0.99], help="Discount factor values")
    parser.add_argument("--adaptive_gamma", action="store_true", default=False, help="Adaptive gamma")
    parser.add_argument("--alpha_actor_values", type=float, nargs="+", default=[0.001], help="Learning rate for actor")
    parser.add_argument("--alpha_critic_values", type=float, nargs="+", default=[0.01], help="Learning rate for critic")
    parser.add_argument("--init", type=float, default=0, help="Initial value of weights")
    parser.add_argument("--episodes_eval", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--eval_steps", type=int, default=500, help="Steps between evaluations")
    parser.add_argument("--max_steps", type=int, default=1000000, help="Maximum training steps")
    parser.add_argument("--n_seeds", type=int, default=30, help="Number of random seeds")
    parser.add_argument('--save_dir', type=str, help="Directory to save the plot")
    parser.add_argument('--experiment_name', type=str, help="Experiment name")
    args = parser.parse_args()

    if args.adaptive_gamma:
        args.gamma_values = [0.1]

    os.makedirs(args.save_dir, exist_ok=True)

    env_id = "Pendulum-v1"
    env = gymnasium.make(env_id)
    env_eval = gymnasium.make(env_id)

    print("Running", env_id, "with", args.init, "as initial weight value!")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # automatically set centers and sigmas
    n_centers = [7] * state_dim
    state_low = env.observation_space.low
    state_high = env.observation_space.high
    centers = np.array(
        np.meshgrid(*[
            np.linspace(
                state_low[i] - (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
                state_high[i] + (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
                n_centers[i],
            )
            for i in range(state_dim)
        ])
    ).reshape(state_dim, -1).T
    sigmas = (state_high - state_low) / np.asarray(n_centers) * 0.99 + 1e-8  # change sigmas for more/less generalization
    get_phi = lambda state : rbf_features(state.reshape(-1, state_dim), centers, sigmas)  # reshape because feature functions expect shape (N, S)
    phi_dummy = get_phi(env.reset()[0])  # to get the number of features

    results_exp_ret = {}
    results_td_err = {}
    results = {}

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    axs[0].set_title("Actor-Critic with adaptive discount factors")
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Expected Return")
    axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)
    axs[0].minorticks_on()

    axs[1].set_title("TD Error")
    axs[1].set_xlabel("Steps")
    axs[1].set_ylabel("TD Error")
    axs[1].grid(True, which="both", linestyle="--", linewidth=0.5)
    axs[1].minorticks_on()

    linestyles = ["-", "--", "-.", ":"]
    colorblind_colors = sns.color_palette("colorblind", len(args.gamma_values))
    plt.rc('axes', prop_cycle=cycler('color', colorblind_colors))
    colors = sns.color_palette("colorblind", len(args.gamma_values))

    for gamma_idx, gamma in enumerate(args.gamma_values):
        color = colors[gamma_idx]
        for alpha_idx, (alpha_actor, alpha_critic) in enumerate(itertools.product(args.alpha_actor_values, args.alpha_critic_values)):
            linestyle = linestyles[alpha_idx % len(linestyles)]
            label = f"γ={gamma}"
            key = (gamma, alpha_actor, alpha_critic)
            results_exp_ret[key] = np.zeros((args.n_seeds, args.max_steps))
            results_td_err[key] = np.zeros((args.n_seeds, args.max_steps))
            train_returns, train_lengths = [], []
            eval_returns, eval_lengths = [], []
            for seed in range(args.n_seeds):
                exp_return_history, train, eval, td_error_history = actor_critic(gamma, args.init, seed, alpha_actor, alpha_critic,  args.episodes_eval, args.eval_steps, args.max_steps, args.adaptive_gamma)
                results_exp_ret[key][seed] = exp_return_history
                results_td_err[key][seed] = td_error_history
                train_returns.append(train[0])
                train_lengths.append(train[1])
                eval_returns.append(eval[0])
                eval_lengths.append(eval[1])
                print(f"γ={gamma}, α_actor={alpha_actor}, α_critic={alpha_critic}, seed={seed}")
            results[gamma] = {
                'training returns': train_returns,
                'training lengths': train_lengths,
                'evaluation returns': eval_returns,
                'evaluation lengths': eval_lengths    
            }
            error_shade_plot(
                axs[0],
                results_exp_ret[key],
                stepsize=1,
                smoothing_window=20,
                label=label,
                linestyle=linestyle,
                color=color
            )
            error_shade_plot(
                axs[1],
                results_td_err[key],
                stepsize=1,
                smoothing_window=20,
                label=label,
                linestyle=linestyle,
                color=color
            )
        gamma_results_path = os.path.join(args.save_dir, f"{args.experiment_name}_init_{args.init}_gamma_{gamma}_results.pkl")
        with open(gamma_results_path, 'wb') as f:
            pickle.dump(results[gamma], f)
        print(f"Gamma-specific results saved to {gamma_results_path}")

    axs[0].legend(fontsize="small", loc="best")
    axs[1].legend(fontsize="small", loc="best")
    plt.tight_layout()

    plot_path = os.path.join(args.save_dir, f"{args.experiment_name}_{args.init}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    # plt.show()

    print(f"Plot saved to {plot_path}")