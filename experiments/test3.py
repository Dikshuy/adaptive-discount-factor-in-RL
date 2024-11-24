import gymnasium as gym
import numpy as np

class RBFApproximator:
    def __init__(self, state_dim, num_features, sigma=1.0):
        self.state_dim = state_dim
        self.num_features = num_features
        self.centers = np.random.uniform(-1, 1, (num_features, state_dim))  # Randomly initialize centers
        self.sigma = sigma

    def transform(self, state):
        """Transform the state into RBF features."""
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        distances = np.linalg.norm(state - self.centers, axis=1) ** 2
        return np.exp(-distances / (2 * self.sigma ** 2))

class ActorCriticRBF:
    def __init__(self, num_features, action_dim, actor_lr=0.01, critic_lr=0.1, gamma=0.99):
        self.num_features = num_features
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma

        # Linear weights for actor and critic
        self.actor_weights = np.random.rand(num_features, action_dim) * 0.01
        self.critic_weights = np.random.rand(num_features) * 0.01

    def get_policy(self, features):
        """Compute softmax policy."""
        preferences = features @ self.actor_weights
        exp_preferences = np.exp(preferences - np.max(preferences))  # Avoid numerical instability
        return exp_preferences / np.sum(exp_preferences)

    def choose_action(self, features):
        """Sample an action based on the policy."""
        policy = self.get_policy(features)
        return np.random.choice(self.action_dim, p=policy)

    def update(self, features, action, reward, next_features, done):
        """Perform Actor-Critic update."""
        # Compute TD target and TD error
        value = features @ self.critic_weights
        next_value = 0 if done else next_features @ self.critic_weights
        td_target = reward + self.gamma * next_value
        td_error = td_target - value

        # Update critic weights
        self.critic_weights += self.critic_lr * td_error * features

        # Update actor weights
        policy = self.get_policy(features)
        grad_log_policy = -policy
        grad_log_policy[action] += 1
        self.actor_weights += self.actor_lr * td_error * np.outer(features, grad_log_policy)

def train_actor_critic_rbf(env_name="CartPole-v1", episodes=50000, num_features=100, sigma=1.0):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize RBF approximator
    rbf_approximator = RBFApproximator(state_dim, num_features, sigma)

    # Initialize Actor-Critic agent
    agent = ActorCriticRBF(num_features, action_dim)

    for episode in range(episodes):
        state, _ = env.reset()
        features = rbf_approximator.transform(state)
        total_reward = 0

        while True:
            action = agent.choose_action(features)
            next_state, reward, done, _, _ = env.step(action)
            next_features = rbf_approximator.transform(next_state)

            # Train the agent
            agent.update(features, action, reward, next_features, done)

            features = next_features
            total_reward += reward

            if done:
                break

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    train_actor_critic_rbf()
