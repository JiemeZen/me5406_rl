from collections import deque
import random

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.num_experiences = 0
        self.buffer = deque(maxlen=buffer_size)

    def get_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        self.buffer.append(experience)

    def clear(self):
        self.buffer.clear()