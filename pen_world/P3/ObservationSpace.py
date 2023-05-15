import numpy as np
import gymnasium as gym

class ObservationSpace(gym.spaces.Space):
    def __init__(self, positions):
        """
        Contructor for space of observations in PenWorld.

        Args:
            positions (array): array of positions in the environment
        """
        self.positions = positions
        self.rng = np.random.default_rng()
        
        self._shape = 2
        self.dtype = float
        self._np_random = False


    def sample(self):
        """
        Return a random position from this space.

        Returns:
            position (array): position in the environment
        """
        return self.positions(self.rng.integers(len(self.positions)))


    def contains(self, x):
        """
        Returns if position is in this space.

        Args:
            x (array): possible position in the environment

        Returns:
            contains (bool): if position is in this space
        """
        return np.all(np.isin(x, self.positions))


    def seed(self, seed: int):
        """
        Seed the random number generator.

        Args:
            seed (int): seed for random number generator
        """
        self.rng = np.random.default_rng(seed)

        
    def to_jsonable(self, sample_n):
        """
        Convert a batch of samples from this space to a JSONable data type.

        Args:
            sample_n (array): array of positions

        Returns:
            list: list of positions
        """
        return sample_n.tolist()
    

    def from_jsonable(self, sample_n):
        """
        Convert a JSONable data type to a batch of samples from this space.

        Args:
            sample_n (list): list of positions 

        Returns:
            array: array of positions
        """
        return np.array(sample_n)