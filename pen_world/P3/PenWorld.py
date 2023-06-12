import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, List
from queue import Queue
from pen_world.P3.PenroseP3 import generate_tiling
from pen_world.P3.ObservationSpace import ObservationSpace
import pygame


REWARD_COORDINATES = (0, 0)
class PenWorld(gym.Env):
    """Gymnasium environment for Penrose environment"""
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, render_mode=None, ngen=5, n=5):
        """
        Generate an environment with a penrose tiling

        Args:
            ngen (int): number of inflation steps to generate penrose env.
            n (int): radius of initialization positions (all initializations are done n steps from env center)
        """
        self.window_size = 512  # The size of the PyGame window


        self.n = n
        self.ngen = ngen
        self.graph = generate_tiling(ngen)
        self._generate_sub_env(n)  # create sub environment of nodes within "n" steps of origin
        self.mask_map, self.action_dim = self.setup_mask_map()  # generate masks for possible action spaces
        self.max_ep_length = self._calculate_max_ep_length()

        # get max x and y positions for rendering
        self.max_x = max([vertex[0] for vertex in self.graph])
        self.max_y = max([vertex[1] for vertex in self.graph])

        self.action_space = spaces.Discrete(self.action_dim)

        self.observation_space = ObservationSpace(self.get_all_positions())
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
    
    def get_all_positions(self):
        return np.array(list(self.graph.keys()))

    def _to_canvas_coordinates(self, env_coordinates):
        """
        Convert environment coordinates to canvas coordinates

        Args:
            env_coordinates (tuple): coordinates in environment

        Returns:
            tuple: coordinates in canvas
        """
        
        # get max x and y coordinates of environment

        env_x, env_y = env_coordinates
        canvas_x = env_x * ((self.window_size / 2) / self.max_x) + self.window_size / 2
        canvas_y = env_y * ((self.window_size / 2) / self.max_y) + self.window_size / 2

        return (canvas_x, canvas_y)


    def _generate_sub_env(self, n):
        """
        Generate sub environment within environment of "radius" n (every vertex is max n steps from origin)

        """
        self.sub_graph_nodes = set()
        queue = Queue()
        origin = (0, 0)

        self.sub_graph_nodes.add(origin)
        init_radius = 0
        queue.put((origin, init_radius))

        while not queue.empty():
            node, radius  = queue.get()
            for neighbor in self.graph[node]:
                neighbor = tuple(neighbor)
                new_radius = radius + 1
                if neighbor not in self.sub_graph_nodes and new_radius <= n:
                    queue.put((neighbor, new_radius))
                    self.sub_graph_nodes.add(neighbor)
                    

    def setup_mask_map(self):
        """
        Generate map from number of actions to mask array which is used by agent to mask invalid actions

        Returns:
            dict[int, array] - dictionary from number of actions to associated mask 
        """
        # find how many distinct option sets there are
        option_set = set()
        for vertex in self.graph:
            option_set.add(len(self.graph[vertex]))

        total_options = sum(option_set)
        start_index = 0
        mask_map = {}
        for num_actions in option_set:
            # create masks that are mapped to the number of actions that are possible
            mask = np.ones(shape=total_options)
            mask[start_index:start_index + num_actions] = 0
            mask_map[num_actions] = mask
            start_index += num_actions

        return mask_map, total_options


    def get_action_mask(self) -> np.ndarray:
        """
        Generate action mask for masking probabilities of agent taking actions since action space varies at different states

        Returns: 
            action_mask - a binary array of possible actions to take at this point

        """

        num_actions = len(self.graph[self.coordinates])

        return self.mask_map[num_actions]


    def _calculate_max_ep_length(self) -> int:
        """
        Calculate the maximum number of steps per episode

        Returns:
            (int): maximum episode length
        """
        num_nodes = len(self.sub_graph_nodes)
        print(f"number of nodes in sub environment: {num_nodes}")
        print(f"max_ep_length: {num_nodes * 2}")
        
        # heuristic: twice the number of nodes in the sub environment
        return num_nodes * 2

    
    def _get_info(self):
        info_dict = {'action_mask': self.get_action_mask()}
        return info_dict
    

    def reset(self, seed=None, options=None):
        """
        Reset the environment position and time step

        Args:
            seed (int, optional): Seed for random number generator used for new position. Defaults to None.
            options (dict, optional): dictionary which can reset to hardcoded new position. Defaults to None.

        Returns:
            observation: position of agent
            info: auxilliary info
        """
        super().reset(seed=seed)
        self.t = 0

        if options is None:
            start_coordinates = None
        else:
            start_coordinates = options['start_coordinates']

        # initialize starting coordinates to random if no specific starting position is given
        if start_coordinates is None:
            self.coordinates = tuple(self.np_random.choice(list(self.sub_graph_nodes))) 
        else: 
            self.coordinates = start_coordinates

        observation = np.array(self.coordinates)
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.trajectory = np.zeros((self.max_ep_length + 1, 2))
        self.trajectory[0] = self.coordinates

        return observation, info


    def step(self, action: int):
        """
        Take a step in the environment

        Args:
            action (int): action corresponding to an edge to traverse in the graph

        Returns:
            observation (array): new position of agent as an (x, y) array
            reward (int): reward for taking action
            terminated (bool): whether or not the episode is terminated (agent reached reward)
            truncated (bool): whether or not the episode is truncated (maximum episode length reached)
            info (dict): auxilliary info
        """
        
        if action < len(self.graph[self.coordinates]):
            self.coordinates = tuple(self.graph[self.coordinates][action])
        else:
            print("WRONG ACTION NUM!")

        # set rewards
        if self.coordinates == REWARD_COORDINATES:
            reward = 1
            terminated = True
        else:
            reward = 0
            terminated = False

        # check if maximum episode length is reached
        self.t += 1
        if self.t == self.max_ep_length:
            truncated = True
        else:
            truncated = False

        if self.render_mode == "human":
            self._render_frame()

        # return state or coordinates
        observation = np.array(self.coordinates)
        info = self._get_info()
        self.trajectory[self.t] = self.coordinates

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create canvas to draw on.
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw the graph
        for point, neighbors in self.graph.items():
            for neighbor in neighbors:
                pygame.draw.line(canvas, (0, 0, 0), self._to_canvas_coordinates(point), self._to_canvas_coordinates(neighbor), width=3)

        agent_size = 5
        # Draw the reward
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._to_canvas_coordinates(REWARD_COORDINATES),
            agent_size,
        )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 59, 0),
            self._to_canvas_coordinates(self.coordinates),
            agent_size,
        )
        
        # Draw the trajectory
        for i in range(self.t)[:-1]:
            pygame.draw.line(canvas, (255, 0, 0), self._to_canvas_coordinates(self.trajectory[i]), self._to_canvas_coordinates(self.trajectory[i+1]), width=3)


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()