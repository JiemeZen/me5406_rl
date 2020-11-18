from collections import deque
import random

class ReplayBuffer():
    """
    Experience replay buffer for DDPG off-policy algorithm.
    """
    def __init__(self, buffer_size):
        """
        Initialize number of experiences and a buffer list. Deque is a FIFO data
        structure which is useful to maintain a fixed array length.
        """
        self.num_experiences = 0
        self.buffer = deque(maxlen=buffer_size)

    def get_batch(self, batch_size):
        """
        Randomly sample a batch of (batch_size)
        """
        return random.sample(self.buffer, batch_size)

    def size(self):
        """
        Current size of the buffer.
        """
        return len(self.buffer)

    def add(self, state, action, reward, new_state, done):
        """
        Append a single experience to the buffer.
        """
        experience = (state, action, reward, new_state, done)
        self.buffer.append(experience)

    def clear(self):
        """
        Erase the buffer.
        """
        self.buffer.clear()