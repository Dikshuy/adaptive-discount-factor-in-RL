import os
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import pickle
import gymnasium as gym
import gym_gridworlds
from envs import get_grid

np.set_printoptions(precision=3)

def eps_greedy_action(env, Q, s, eps):
    if np.random.rand() < eps:
        action = np.random.choice(env.action_space.n)
    else:
        action = np.random.choice(np.where(Q[s] == np.max(Q[s]))[0])
    return action

def expected_return(env, Q, gamma, episodes=1):
    G = np.zeros(episodes)
    episode_steps = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=int(seed))
        done = False
        t = 0
        while not done:
            a = eps_greedy_action(env, Q, s, 0.0)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G[e] += gamma**t * r
            s = s_next
            t += 1
            if done:
                episode_steps[e] = t
    return G.mean(), episode_steps.mean()

def Q_learning(env, env_eval, Q, gamma, gamma_env, eps, alpha, max_steps, eval_steps, adaptive_gamma, _seed):
    exp_ret = []
    steps_per_episode = []
    eps_decay = eps / max_steps
    tot_steps = 0
    episodes = 0

    while tot_steps < max_steps:  
        s, _ = env.reset(seed = _seed)
        done = False

        while not done and tot_steps < max_steps:
            tot_steps += 1
            a = eps_greedy_action(env, Q, s, eps)
            s_next, r, terminated, truncated, _ = env.step(a)

            done = terminated or truncated
            eps = max(eps - eps_decay, 1e-5)

            td_err = r + gamma * np.max(Q[s_next]) * (1 - terminated) - Q[s, a]

            Q[s,a] += alpha * td_err

            if tot_steps % eval_steps == 0:
                G, episode_steps = expected_return(env_eval, Q, gamma_env)
                exp_ret.append(G)
                steps_per_episode.append(episode_steps)

            if done:
                episodes += 1
                if adaptive_gamma:
                    gamma = min(gamma + episodes * gamma / max_steps, 1.0)

            s = s_next

    return Q, exp_ret, steps_per_episode

def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span:])
    return re

def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)

    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Learning for SimpleGrid")
    parser.add_argument("--environments", type=str, nargs="+", required=True, help="List of environment names (e.g., EASY_SPARSE, DIFFICULT_DENSE)")
    parser.add_argument("--gamma_values", type=float, nargs="+", default=[0.1, 0.5, 0.9], help="Gamma values")
    parser.add_argument("--adaptive_gamma", action="store_true", default=False, help="Adaptive gamma")
    parser.add_argument("--alpha_values", type=float, nargs="+", default=[0.1], help="Alpha values")
    parser.add_argument("--initial_values", type=float, nargs="+", default=[0.0, 10.0], help="Initial Q values")
    parser.add_argument("--max_steps", type=int, default=300000, help="Max steps for training")
    parser.add_argument("--eval_steps", type=int, default=100, help="Steps between evaluations")
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of random seeds")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save plots")
    args = parser.parse_args()

    gamma_env = 0.99

    if args.adaptive_gamma:
        args.gamma_values = [0.1]

    linestyles = ["-", "--", "-.", ":"]
    colorblind_colors = sns.color_palette("colorblind", len(args.gamma_values))
    plt.rc('axes', prop_cycle=cycler('color', colorblind_colors))
    colors = sns.color_palette("colorblind", len(args.gamma_values))

    output_dirs = {}
    for env_name in args.environments:
        output_dir = os.path.join(args.save_dir, env_name)
        os.makedirs(output_dir, exist_ok=True)
        output_dirs[env_name] = output_dir

    for env_name in args.environments:
        env = gym.make(env_name)
        env_eval = gym.make(env_name)

        print("Loading:", env_name, "grid")

        results_exp_ret = np.zeros((
            len(args.gamma_values),
            len(args.alpha_values),
            len(args.initial_values),
            args.n_seeds,
            args.max_steps // args.eval_steps,
        ))

        results_steps = np.zeros((
            len(args.gamma_values),
            len(args.alpha_values),
            len(args.initial_values),
            args.n_seeds,
            args.max_steps // args.eval_steps,
        ))

        fig, axs = plt.subplots(1, 2, figsize=(12, 8))

        for ax in axs:
            ax.set_xlabel("Steps (X100)", fontsize=10)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.minorticks_on()

        fig1, axs1 = plt.subplots(1, 1, figsize=(12, 8))
        axs1.set_xlabel("Steps(X100)")
        axs1.set_ylabel("Expected Return")
        axs1.set_ylim([-5, 1.4])
        axs1.grid(True, which="both", linestyle="--", linewidth=0.5)
        axs1.minorticks_on()

        for i, gamma in enumerate(args.gamma_values):
            for j, alpha in enumerate(args.alpha_values):
                for k, init_value in enumerate(args.initial_values):
                    for seed in range(args.n_seeds):
                        Q = np.zeros((env.observation_space.n, env.action_space.n)) + init_value
                        Q, exp_ret, steps = Q_learning(env, env_eval, Q, gamma, gamma_env, 1.0, alpha, args.max_steps, args.eval_steps, args.adaptive_gamma, seed)
                        results_exp_ret[i, j, k, seed] = exp_ret
                        results_steps[i, j, k, seed] = steps

                        print(f"γ={gamma}, Q_o={init_value}, seed={seed}")

                    if args.adaptive_gamma:
                        label = f"γ=adaptive γ"
                    else:
                        label = f"γ={gamma}, $Q_o$={init_value}"

                    color = sns.color_palette("colorblind")[args.gamma_values.index(gamma)]

                    error_shade_plot(
                        axs[0], results_exp_ret[i, j, k], stepsize=1, smoothing_window=20,
                        label=label, linestyle=linestyles[j % len(linestyles)], color=color
                    )
                    error_shade_plot(
                        axs[1], results_steps[i, j, k], stepsize=1, smoothing_window=20,
                        label=label, linestyle=linestyles[j % len(linestyles)], color=color
                    )

                    error_shade_plot(
                        axs1, results_exp_ret[i, j, k], stepsize=1, smoothing_window=20,
                        label=label, linestyle=linestyles[j % len(linestyles)], color=color
                    )

                    sub_dir = os.path.join(
                        output_dirs[env_name],
                        f"gamma_{gamma}",
                        f"Q_init_{init_value}"
                    )
                    os.makedirs(sub_dir, exist_ok=True)

                    gamma_results_path = os.path.join(sub_dir, f"{gamma}_results.pkl")
                    with open(gamma_results_path, 'wb') as f:
                        pickle.dump({
                            "exp_ret": results_exp_ret[i, j, k],
                            "steps": results_steps[i, j, k]
                        }, f)
                    print(f"Gamma-specific results saved to {gamma_results_path}")

        axs[0].set_ylabel("Expected Return", fontsize=10)
        axs[0].set_ylim([-5, 1.4])
        if args.adaptive_gamma:
            axs[0].set_title("Q-Learning Performance with adaptive γ")
            axs1.set_title("Q-Learning Performance with adaptive γ")
        else:
            axs[0].set_title("Q-Learning Performance")
            axs1.set_title("Q-Learning Performance")
        axs[0].legend(fontsize="small", loc="best")
        axs1.legend(fontsize="small", loc="best")

        axs[1].set_ylabel("Steps to Goal", fontsize=10)
        if args.adaptive_gamma:
            axs[1].set_title("Steps per Episode with adaptive γ")
        else:
            axs[1].set_title("Steps per Episode")
        axs[1].legend(fontsize="small", loc="best")

        if args.adaptive_gamma:
            plot_path = os.path.join(output_dirs[env_name], f"adaptive_γ_q_learning_e_{args.initial_values}.png")
            fig.savefig(plot_path, dpi=300)
            plot_path = os.path.join(output_dirs[env_name], f"adaptive_γ_q_learning_{args.initial_values}.png")
            fig1.savefig(plot_path, dpi=300)
        else:
            plot_path = os.path.join(output_dirs[env_name], f"q_learning_e_{args.initial_values}.png")
            fig.savefig(plot_path, dpi=300)
            plot_path = os.path.join(output_dirs[env_name], f"q_learning_{args.initial_values}.png")
            fig1.savefig(plot_path, dpi=300)
        # plt.show()
        plt.close()
