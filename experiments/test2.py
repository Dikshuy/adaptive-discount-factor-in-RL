import numpy as np
import gymnasium as gym

# Set random seed for reproducibility
np.random.seed(0)

def create_rbf_features(num_centers, state_low, state_high):
    """
    Create evenly spaced RBF centers and compute widths.
    :param num_centers: Number of RBF centers per dimension
    :param state_low: Lower bound of the state space
    :param state_high: Upper bound of the state space
    :return: RBF centers and widths
    """
    for i, state in enumerate(state_high):
        if state == np.inf: state_high[i] = 10000000000000000000000
    for i, state in enumerate(state_low):
        if state == -np.inf: state_low[i] = -10000000000000000000000

    centers = [np.linspace(state_low[i], state_high[i], num_centers) for i in range(len(state_low))]
    centers = np.array(np.meshgrid(*centers)).reshape(len(state_low), -1).T
    widths = (state_high - state_low) / (num_centers - 1) / 2
    return centers, widths

def rbf_features(state, centers, widths):
    """
    Compute RBF features for a given state.
    :param state: The current state
    :param centers: RBF centers
    :param widths: RBF widths
    :return: Feature vector
    """
    diff = state - centers
    return np.exp(-np.sum((diff / widths) ** 2, axis=1))

# Environment setup
env = gym.make("CartPole-v1")
state_low = env.observation_space.low
state_high = env.observation_space.high
num_actions = env.action_space.n

# RBF parameters
num_rbf_centers = 5  # Number of centers per dimension
centers, widths = create_rbf_features(num_rbf_centers, state_low, state_high)
num_features = centers.shape[0]

# Hyperparameters
alpha = 0.01  # Learning rate for the actor
beta = 0.05   # Learning rate for the critic
gamma = 0.99  # Discount factor

# Initialize weights for actor and critic
actor_weights = np.zeros((num_features, num_actions))
critic_weights = np.zeros(num_features)

def policy(features):
    """Compute the softmax policy."""
    preferences = np.dot(features, actor_weights)
    exp_preferences = np.exp(preferences - np.max(preferences))
    return exp_preferences / np.sum(exp_preferences)

def value(features):
    """Compute the value function as a linear approximation."""
    return np.dot(features, critic_weights)

# Training loop
num_episodes = 500000
for episode in range(num_episodes):
    state, _ = env.reset()
    features = rbf_features(state, centers, widths)
    done = False
    episode_reward = 0

    while not done:
        # Choose action based on policy
        probs = policy(features)
        action = np.random.choice(num_actions, p=probs)

        # Execute the action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_features = rbf_features(next_state, centers, widths)

        # TD error
        td_target = reward + (gamma * value(next_features) if not terminated else 0)
        td_error = td_target - value(features)

        # Update critic weights
        critic_weights += beta * td_error * features

        # Update actor weights
        for a in range(num_actions):
            actor_weights[:, a] -= alpha * td_error * probs[a] * features
        actor_weights[:, action] += alpha * td_error * features

        # Update state and features
        features = next_features
        episode_reward += reward

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

env.close()
