from __future__ import annotations
import logging
import numpy as np
from gymnasium import spaces, Env
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

MAPS = {
    "4x4": ["0000", "0201", "0001", "1000"],  # Added lava (2) to the default map
    "8x8": [
        "00000000",
        "00000000",
        "00010000",
        "00002100",
        "00010000",
        "01100210",
        "01001010",
        "00010000",
    ],
}


class SimpleGridEnv(Env):
    """
    Simple Grid Environment with Lava

    The environment is a grid with obstacles (walls), lava, and agents. The agents can move in 
    one of the four cardinal directions. If they try to move over an obstacle or out of the grid 
    bounds, they stay in place. Stepping on lava tiles gives a -5 reward. Each agent has a unique 
    color and a goal state of the same color.

    Grid cell types:
    0: Free cell
    1: Wall/Obstacle
    2: Lava (dangerous tile with -5 reward)
    """
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], 'render_fps': 8}
    FREE: int = 0
    OBSTACLE: int = 1
    LAVA: int = 2  # New tile type for lava
    MOVES: dict[int,tuple] = {
        0: (-1, 0), #UP
        1: (1, 0),  #DOWN
        2: (0, -1), #LEFT
        3: (0, 1)   #RIGHT
    }

    def __init__(self,     
        obstacle_map: str | list[str],
        render_mode: str | None = None,
        stochasticity = 0.0,
    ):
        """
        Initialise the environment.

        Parameters
        ----------
        obstacle_map: str | list[str]
            Map to be loaded. If a string is passed, the map is loaded from pre-existing maps.
            The map format now supports three digits:
            0: Free cell
            1: Wall
            2: Lava
        """
        # Env configuration
        self.obstacles = self.parse_obstacle_map(obstacle_map)
        self.nrow, self.ncol = self.obstacles.shape

        self.action_space = spaces.Discrete(len(self.MOVES))
        self.observation_space = spaces.Discrete(n=self.nrow*self.ncol)
        
        # Rendering configuration
        self.fig = None
        self.render_mode = render_mode
        self.fps = self.metadata['render_fps']
        
        self.stochasticity = stochasticity

    def reset(
            self, 
            seed: int | None = None, 
            options: dict = dict()
        ) -> tuple:
        """
        Reset the environment.

        Parameters
        ----------
        seed: int | None
            Random seed.
        options: dict
            Optional dict that allows you to define the start (`start_loc` key) and goal (`goal_loc`key) position when resetting the env. By default options={}, i.e. no preference is expressed for the start and goal states and they are randomly sampled.
        """

        # Set seed
        super().reset(seed=seed)

        # parse options
        self.start_xy = self.parse_state_option('start_loc', options)
        self.goal_xy = self.parse_state_option('goal_loc', options)

        # initialise internal vars
        self.agent_xy = self.start_xy
        self.reward = self.get_reward(*self.agent_xy)
        self.done = self.on_goal()
        self.agent_action = None
        self.n_iter = 0

        # Check integrity
        self.integrity_checks()

        #if self.render_mode == "human":
        self.render()

        return self.get_obs(), self.get_info()
    
    def step(self, action: int):
        """
        Take a step in the environment.
        """
        #assert action in self.action_space
        self.agent_action = action
        
        if np.random.rand() < self.stochasticity:
            action = np.random.randint(0, len(self.MOVES))
            self.agent_action = action
            
        # Get the current position of the agent
        row, col = self.agent_xy
        dx, dy = self.MOVES[action]

        # Compute the target position of the agent
        target_row = row + dx
        target_col = col + dy
        
        aux_reward = self.agent_action == 0
        aux_reward += self.agent_action == 3
        
        # Compute the reward
        self.reward = self.get_reward(target_row, target_col) + (aux_reward * 0.1)
        
        # Check if the move is valid
        if self.is_in_bounds(target_row, target_col) and self.is_free(target_row, target_col):
            self.agent_xy = (target_row, target_col)
            self.done = self.on_goal()

        self.n_iter += 1

        # if self.render_mode == "human":
        self.render()
        
        truncated = self.n_iter > 100

        return self.get_obs(), self.reward, self.done, truncated, self.get_info()
    
    def parse_obstacle_map(self, obstacle_map) -> np.ndarray:
        """
        Initialise the grid.

        The grid is described by a map, i.e. a list of strings where each string denotes a row of the grid and is a sequence of 0s and 1s, where 0 denotes a free cell and 1 denotes a wall cell.

        The grid can be initialised by passing a map name or a custom map.
        If a map name is passed, the map is loaded from a set of pre-existing maps. If a custom map is passed, the map provided by the user is parsed and loaded.

        Examples
        --------
        >>> my_map = ["001", "010", "011]
        >>> SimpleGridEnv.parse_obstacle_map(my_map)
        array([[0, 0, 1],
               [0, 1, 0],
               [0, 1, 1]])
        """
        if isinstance(obstacle_map, list):
            map_str = np.asarray(obstacle_map, dtype='c')
            map_int = np.asarray(map_str, dtype=int)
            return map_int
        elif isinstance(obstacle_map, str):
            map_str = MAPS[obstacle_map]
            map_str = np.asarray(map_str, dtype='c')
            map_int = np.asarray(map_str, dtype=int)
            return map_int
        else:
            raise ValueError(f"You must provide either a map of obstacles or the name of an existing map. Available existing maps are {', '.join(MAPS.keys())}.")
        
    def parse_state_option(self, state_name: str, options: dict) -> tuple:
        """
        parse the value of an option of type state from the dictionary of options usually passed to the reset method. Such value denotes a position on the map and it must be an int or a tuple.
        """
        try:
            state = options[state_name]
            if isinstance(state, int):
                return self.to_xy(state)
            elif isinstance(state, tuple):
                return state
            else:
                raise TypeError(f'Allowed types for `{state_name}` are int or tuple.')
        except KeyError:
            state = self.sample_valid_state_xy()
            logger = logging.getLogger()
            logger.info(f'Key `{state_name}` not found in `options`. Random sampling a valid value for it:')
            logger.info(f'...`{state_name}` has value: {state}')
            return state

    def sample_valid_state_xy(self) -> tuple:
        state = self.observation_space.sample()
        pos_xy = self.to_xy(state)
        while not self.is_free(*pos_xy):
            state = self.observation_space.sample()
            pos_xy = self.to_xy(state)
        return pos_xy
    
    def integrity_checks(self) -> None:
        # check that goals do not overlap with walls
        assert self.obstacles[self.start_xy] == self.FREE, \
            f"Start position {self.start_xy} overlaps with a wall."
        assert self.obstacles[self.goal_xy] == self.FREE, \
            f"Goal position {self.goal_xy} overlaps with a wall."
        assert self.is_in_bounds(*self.start_xy), \
            f"Start position {self.start_xy} is out of bounds."
        assert self.is_in_bounds(*self.goal_xy), \
            f"Goal position {self.goal_xy} is out of bounds."
        
    def to_s(self, row: int, col: int) -> int:
        """
        Transform a (row, col) point to a state in the observation space.
        """
        return row * self.ncol + col

    def to_xy(self, s: int) -> tuple[int, int]:
        """
        Transform a state in the observation space to a (row, col) point.
        """
        return (s // self.ncol, s % self.ncol)

    def on_goal(self) -> bool:
        """
        Check if the agent is on its own goal.
        """
        return self.agent_xy == self.goal_xy
    
    def is_in_bounds(self, row: int, col: int) -> bool:
        """
        Check if a target cell is in the grid bounds.
        """
        return 0 <= row < self.nrow and 0 <= col < self.ncol

    def get_reward(self, x: int, y: int) -> float:
        """
        Get the reward of a given cell.
        """
        if not self.is_in_bounds(x, y):
            return 0.0
        elif not self.is_free(x, y):
            return 0.0  # Wall
        if (x, y) == self.goal_xy:
            return 10.0
        elif (self.obstacles[x, y] == self.LAVA):  # Check for lava
            return -1.0
        else:
            return 0.0

    def is_free(self, row: int, col: int) -> bool:
        """
        Check if a cell is free (not a wall).
        Note: Lava tiles are considered "free" for movement purposes
        but have a negative reward.
        """
        return self.obstacles[row, col] != self.OBSTACLE

    def get_obs(self) -> int:
        return self.to_s(*self.agent_xy)
    
    def get_info(self) -> dict:
        return {
            'agent_xy': self.agent_xy,
            'n_iter': self.n_iter,
        }

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode is None:
            return None
        
        elif self.render_mode == "ansi":
            s = f"{self.n_iter},{self.agent_xy[0]},{self.agent_xy[1]},{self.reward},{self.done},{self.agent_action}\n"
            #print(s)
            return s

        elif self.render_mode == "rgb_array":
            self.render_frame()
            self.fig.canvas.draw()
            img = np.array(self.fig.canvas.renderer.buffer_rgba())
            return img
    
        elif self.render_mode == "human":
            self.render_frame()
            plt.pause(1/self.fps)
            return None
        
        else:
            raise ValueError(f"Unsupported rendering mode {self.render_mode}")

    def render_frame(self):
        if self.fig is None:
            self.render_initial_frame()
            self.fig.canvas.mpl_connect('close_event', self.close)
        else:
            self.update_agent_patch()
        self.ax.set_title(f"Step: {self.n_iter}, Reward: {self.reward}")
    
    def create_agent_patch(self):
        """
        Create a Circle patch for the agent.

        @NOTE: If agent position is (x,y) then, to properly render it, we have to pass (y,x) as center to the Circle patch.
        """
        return mpl.patches.Circle(
            (self.agent_xy[1]+.5, self.agent_xy[0]+.5), 
            0.3, 
            facecolor='orange', 
            fill=True, 
            edgecolor='black', 
            linewidth=1.5,
            zorder=100,
        )

    def update_agent_patch(self):
        """
        @NOTE: If agent position is (x,y) then, to properly 
        render it, we have to pass (y,x) as center to the Circle patch.
        """
        self.agent_patch.center = (self.agent_xy[1]+.5, self.agent_xy[0]+.5)
        return None
    
    def render_initial_frame(self):
        """
        Render the initial frame.

        @NOTE: 
        0: free cell (white)
        1: obstacle (black)
        2: lava (red)
        3: start position (blue)
        4: goal (green)
        """
        data = self.obstacles.copy()
        
        # Temporarily store the lava positions
        self.lava_positions = np.where(data == self.LAVA)
        
        # Set start and goal positions (using higher numbers to distinguish from lava)
        data[self.start_xy] = 3
        data[self.goal_xy] = 4

        colors = ['white', 'black', 'red', 'blue', 'green']
        bounds = [i-0.1 for i in range(6)]  # Adjusted for 5 different values

        # create discrete colormap
        cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        plt.ion()
        fig, ax = plt.subplots(tight_layout=True)
        self.fig = fig
        self.ax = ax

        ax.grid(axis='both', color='k', linewidth=1.3) 
        ax.set_xticks(np.arange(0, data.shape[1], 1))
        ax.set_yticks(np.arange(0, data.shape[0], 1))
        ax.tick_params(
            bottom=False, 
            top=False, 
            left=False, 
            right=False, 
            labelbottom=False, 
            labelleft=False
        ) 

        # draw the grid
        ax.imshow(
            data, 
            cmap=cmap, 
            norm=norm,
            extent=[0, data.shape[1], data.shape[0], 0],
            interpolation='none'
        )

        # Create white holes on start and goal positions
        for pos in [self.start_xy, self.goal_xy]:
            wp = self.create_white_patch(*pos)
            ax.add_patch(wp)

        # Create agent patch in start position
        self.agent_patch = self.create_agent_patch()
        ax.add_patch(self.agent_patch)

        return None

    def create_white_patch(self, x, y):
        """
        Render a white patch in the given position.
        """
        return mpl.patches.Circle(
            (y+.5, x+.5), 
            0.4, 
            color='white', 
            fill=True, 
            zorder=99,
        )

    def close(self, *args):
        """
        Close the environment.
        """
        plt.close(self.fig)
        sys.exit()
        
