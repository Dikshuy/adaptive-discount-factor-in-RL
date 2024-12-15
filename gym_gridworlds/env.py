import gymnasium
import gym_gridworlds
import matplotlib.pyplot as plt

environments = ["Empty-10x10-v0", "Empty-Distract-6x6-v0", "Penalty-3x3-v0", "Quicksand-4x4-v0", "Quicksand-Distract-4x4-v0", "TwoRoom-Quicksand-3x5-v0", "Full-4x5-v0", "TwoRoom-Distract-Middle-2x11-v0", "Barrier-5x5-v0"]

for env_name in environments:
    env = gymnasium.make(f"Gym-Gridworlds/{env_name}", render_mode="rgb_array")

    obs, info = env.reset()

    frames = []

    frames.append(env.render())

    plt.imshow(frames[0])
    plt.axis("off")
    plt.savefig(f"gridworlds/{env_name}.png", bbox_inches="tight")
    plt.close()
    print("saved", env_name, "gridworld")
    env.close()
