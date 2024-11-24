import gymnasium
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)

def cantor_pairing(x, y):
    return int(0.5 * (x + y) * (x + y + 1) + y)

def rbf_features(x: np.array, c: np.array, s: np.array) -> np.array:
    return np.exp(-(((x[:, None] - c[None]) / s[None])**2).sum(-1) / 2.0)

def expected_return(env, weights, gamma, episodes=50):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            phi = get_phi(s)
            a = eps_greedy_action(phi, weights, 0)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G[e] += r
            s = s_next
            t += 1
    return G.mean()

def eps_greedy_action(phi, weights, eps):
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    else:
        Q = np.dot(phi, weights).ravel()
        best = np.argwhere(Q == Q.max())
        i = np.random.choice(range(best.shape[0]))
        return best[i][0]
    
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

def actor_critic(gamma, seed):
    actor_weights = np.zeros((phi_dummy.shape[1], n_actions))
    critic_weights = np.zeros(phi_dummy.shape[1])
    eps = 1.0
    eps_decay = eps / max_steps
    tot_steps = 0
    exp_return_history = np.zeros(max_steps)
    exp_return = expected_return(env_eval, actor_weights, gamma, episodes_eval)
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
                exp_return = expected_return(env_eval, actor_weights, gamma, episodes_eval)
            
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


env_id = "CartPole-v1"
env = gymnasium.make(env_id)
env_eval = gymnasium.make(env_id)

state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# automatically set centers and sigmas
n_centers = [20] * state_dim
state_low = env.observation_space.low
state_high = env.observation_space.high

'''
cutting inf to high numbers for RBFs
discuss and change later if required
'''
for i, state in enumerate(state_low):
    if state == -np.inf:
        state_low[i] = -5.5

for i, state in enumerate(state_high):
    if state == np.inf:
        state_high[i] = 5.5

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

# hyperparameters
gamma_values = [0.1, 0.5, 0.8, 0.99]
alpha_actor = 0.001
alpha_critic = 0.01
episodes_eval = 50
eval_steps = 100
max_steps = 20000
n_seeds = 1
results_exp_ret = np.zeros((
    len(gamma_values),
    n_seeds,
    max_steps,
))

fig, axs = plt.subplots(1, 1)
axs.set_prop_cycle(color=["red", "green", "blue", "cyan"])
axs.set_title("Actor-Critic with different discount factors")
axs.set_xlabel("Steps")
axs.set_ylabel("Expected Return")
axs.grid(True, which="both", linestyle="--", linewidth=0.5)
axs.minorticks_on()

for i, gamma in enumerate(gamma_values):
    for seed in range(n_seeds):
        exp_return_history = actor_critic(gamma, int(seed))
        results_exp_ret[i, seed] = exp_return_history
        print(gamma, seed)

    plot_args = dict(
        stepsize=1,
        smoothing_window=20,
        label=gamma,
    )
    error_shade_plot(
        axs,
        results_exp_ret[i],
        **plot_args,
    )
    axs.legend()

plt.savefig("actor_critic_cartpole.png", dpi=300)
plt.show()