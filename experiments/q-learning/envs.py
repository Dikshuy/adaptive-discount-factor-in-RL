import gymnasium as gym
import gym_gridworlds

def get_grid(name):
    grids = ["Gym-Gridworlds/Straight-20-v0",
            "Gym-Gridworlds/Empty-2x2-v0",
            "Gym-Gridworlds/Empty-3x3-v0",
            "Gym-Gridworlds/Empty-Loop-3x3-v0",
            "Gym-Gridworlds/Empty-10x10-v0",
            "Gym-Gridworlds/Empty-Distract-6x6-v0",
            "Gym-Gridworlds/Penalty-3x3-v0",
            "Gym-Gridworlds/Quicksand-4x4-v0",
            "Gym-Gridworlds/Quicksand-Distract-4x4-v0",
            "Gym-Gridworlds/TwoRoom-Quicksand-3x5-v0",
            "Gym-Gridworlds/Corridor-3x4-v0",
            "Gym-Gridworlds/Full-4x5-v0",
            "Gym-Gridworlds/TwoRoom-Distract-Middle-2x11-v0",
            "Gym-Gridworlds/Barrier-5x5-v0",
            "Gym-Gridworlds/RiverSwim-6-v0",
            "Gym-Gridworlds/CliffWalk-4x12-v0",
            "Gym-Gridworlds/DangerMaze-6x6-v0",
    ]

    for grid in grids:
        if name == grid:
            return grid

        else:
            NotImplementedError("grid not found!")