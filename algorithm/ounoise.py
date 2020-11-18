import numpy as np
import numpy.random as nr

class OUNoise:
    """
    Ornstein Uhlenbeck Noise
    """

    def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        """
        Reset the state.
        """
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        """
        Generate Noise. To be added with predicted action to encourage explorations.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state