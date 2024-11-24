import numpy as np
import gymnasium as gym
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler

class RBFApproximator:
    def __init__(self, n_features, env):
        samples = np.array([env.observation_space.sample() for _ in range(10000)])
        self.scaler = StandardScaler()
        self.scaler.fit(samples)
        
        self.rbf = RBFSampler(gamma=0.5, n_components=n_features, random_state=123)
        self.rbf.fit(samples)
        
        self.weights = np.random.randn(n_features)
    
    def transform(self, state):
        state = np.array(state, dtype=np.float32).flatten()  # Ensure state is a flat array
        scaled_state = self.scaler.transform(state.reshape(1, -1))
        return self.rbf.transform(scaled_state)[0]

    def predict(self, state):
        features = self.transform(state)
        return np.dot(features, self.weights)

    def update(self, state, target, lr):
        features = self.transform(state)
        td_error = target - np.dot(features, self.weights)
        self.weights += lr * td_error * features

class ActorCritic:
    def __init__(self, env, n_features=100, gamma=0.99, lr_actor=0.01, lr_critic=0.1):
        self.env = env
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.actor_weights = np.random.randn(n_features, env.action_space.n)
        self.rbf_actor = RBFApproximator(n_features, env)
        self.critic = RBFApproximator(n_features, env)

    def policy(self, state):
        features = self.rbf_actor.transform(state)
        logits = np.dot(features, self.actor_weights)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def select_action(self, state):
        probs = self.policy(state)
        return np.random.choice(len(probs), p=probs)

    def update(self, state, action, reward, next_state, terminated, truncated):
        target = reward + self.gamma * self.critic.predict(next_state) * (1 - terminated)
        self.critic.update(state, target, self.lr_critic)

        td_error = target - self.critic.predict(state)
        features = self.rbf_actor.transform(state)
        probs = self.policy(state)
        grad = -probs
        grad[action] += 1
        self.actor_weights[:, action] += self.lr_actor * td_error * features

def train_cartpole(episodes=2000):
    env = gym.make('CartPole-v1')
    agent = ActorCritic(env)

    for episode in range(episodes):
        state, _ = env.reset()  # Extract only the observation
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, terminated, truncated)

            state = next_state
            total_reward += reward
            if done:
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    train_cartpole()
