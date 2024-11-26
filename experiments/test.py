import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import itertools

np.set_printoptions(precision=3, suppress=True)

def cantor_pairing(x, y):
    return int(0.5 * (x + y) * (x + y + 1) + y)

def rbf_features(x: np.array, c: np.array, s: np.array) -> np.array:
    # print(np.exp(-(((x[:, None] - c[None]) / s[None])**2).sum(-1) / 2.0))
    return np.exp(-(((x[:, None] - c[None]) / s[None])**2).sum(-1) / 2.0)

def expected_return(env, weights, eps, episodes=10):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            phi = get_phi(s)
            a = softmax_action(phi, weights, eps)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G[e] += r
            s = s_next
            t += 1
    return G.mean()

def softmax_probs(phi, weights, eps):
    q = np.dot(phi, weights)
    q_exp = np.exp((q - np.max(q, -1, keepdims=True)) / max(eps, 1e-12))
    probs = q_exp / q_exp.sum(-1, keepdims=True)
    return probs

def softmax_action(phi, weights, eps):
    probs = softmax_probs(phi, weights, eps)
    return np.random.choice(weights.shape[1], p=probs.ravel())

def dlog_softmax_probs(phi, weights, eps, act):
    probs = softmax_probs(phi, weights, eps)
    dlog_pi = phi.T @ (np.eye(n_actions)[act] - probs)
    return dlog_pi

def actor_critic(gamma, seed, alpha_actor, alpha_critic, episodes_eval, eval_steps, max_steps):
    actor_weights = np.zeros((phi_dummy.shape[1], n_actions))
    critic_weights = np.zeros(phi_dummy.shape[1])
    eps = 1.0
    eps_decay = eps / max_steps
    tot_steps = 0
    exp_return_history = np.zeros(max_steps)
    exp_return = expected_return(env_eval, actor_weights, eps, episodes_eval)
    pbar = tqdm(total=max_steps)

    while tot_steps < max_steps:
        s, _ = env.reset(seed = seed)
        done = False
        T = 0

        while not done and tot_steps < max_steps:
            phi = get_phi(s)
            a = softmax_action(phi, actor_weights, eps)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            phi_next = get_phi(s_next)
           
            v = np.dot(phi, critic_weights)
            v_next = np.dot(phi_next, critic_weights)
            
            td_error = r + gamma * v_next * (1 - terminated) - v
            critic_weights += alpha_critic * td_error * phi.flatten()
            dlog_pi = dlog_softmax_probs(phi, actor_weights, eps, a)
            actor_weights += alpha_actor * td_error * dlog_pi

            s = s_next
            T += 1
            tot_steps += 1

            exp_return_history[tot_steps-1] = exp_return

            if tot_steps % eval_steps == 0:
                exp_return = expected_return(env_eval, actor_weights, eps, episodes_eval)
            
            eps = max(eps - eps_decay, 0.01)

            pbar.set_description(
                f"G: {exp_return:.3f}"
        )
        pbar.update(T)

    pbar.close()
    return exp_return_history

def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
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
    parser = argparse.ArgumentParser(description="Actor-Critic for Acrobot-v1")
    parser.add_argument("--gamma_values", type=float, nargs="+", default=[0.1, 0.5, 0.8, 0.99], help="Discount factor values")
    parser.add_argument("--alpha_actor_values", type=float, nargs="+", default=[0.001], help="Learning rate for actor")
    parser.add_argument("--alpha_critic_values", type=float, nargs="+", default=[0.01], help="Learning rate for critic")
    parser.add_argument("--episodes_eval", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--eval_steps", type=int, default=100, help="Steps between evaluations")
    parser.add_argument("--max_steps", type=int, default=20000, help="Maximum training steps")
    parser.add_argument("--n_seeds", type=int, default=30, help="Number of random seeds")
    parser.add_argument('--save_dir', type=str, help="Directory to save the plot")
    parser.add_argument('--experiment_name', type=str, help="Experiment name")
    args = parser.parse_args()
    
    env_id = "Acrobot-v1"
    env = gymnasium.make(env_id)
    env_eval = gymnasium.make(env_id)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # automatically set centers and sigmas
    n_centers = [7]*state_dim
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
    sigmas = (state_high - state_low) / np.asarray(n_centers) * 0.75 + 1e-8  # change sigmas for more/less generalization
    get_phi = lambda state : rbf_features(state.reshape(-1, state_dim), centers, sigmas)  # reshape because feature functions expect shape (N, S)
    phi_dummy = get_phi(env.reset()[0])  # to get the number of features

    results_exp_ret = {}

    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    axs.set_prop_cycle(color=["red", "green", "blue", "cyan"])
    axs.set_title("Actor-Critic with different discount factors")
    axs.set_xlabel("Steps")
    axs.set_ylabel("Expected Return")
    axs.grid(True, which="both", linestyle="--", linewidth=0.5)
    axs.minorticks_on()

    linestyles = ["-", "--", "-.", ":"]
    colors = plt.cm.viridis(np.linspace(0, 1, len(args.gamma_values)))

    for gamma_idx, gamma in enumerate(args.gamma_values):
        color = colors[gamma_idx]
        for alpha_idx, (alpha_actor, alpha_critic) in enumerate(itertools.product(args.alpha_actor_values, args.alpha_critic_values)):
            linestyle = linestyles[alpha_idx % len(linestyles)]
            label = f"γ={gamma}, α_actor={alpha_actor}, α_critic={alpha_critic}"
            key = (gamma, alpha_actor, alpha_critic)
            results_exp_ret[key] = np.zeros((args.n_seeds, args.max_steps))
            for seed in range(args.n_seeds):
                exp_return_history = actor_critic(gamma, seed, alpha_actor, alpha_critic,  args.episodes_eval, args.eval_steps, args.max_steps)
                results_exp_ret[key][seed] = exp_return_history
                print(f"γ={gamma}, α_actor={alpha_actor}, α_critic={alpha_critic}, seed={seed}")
            error_shade_plot(
                axs,
                results_exp_ret[key],
                stepsize=1,
                smoothing_window=20,
                label=label,
                linestyle=linestyle,
                color=color
            )


    axs.legend(fontsize="small", loc="best")
    plt.tight_layout()

    plt.savefig("test.png", dpi=300)
    plt.close()
    # plt.show()