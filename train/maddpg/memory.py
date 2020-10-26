#
# coding=utf-8

from collections import namedtuple
import random

# create class
Experience = namedtuple('Experience',
                        # attrs
                        ('img_states', 'info_states', 'jrev_states', 'actions',
                         'next_img_states', 'next_info_states', 'next_jrev_states', 'rewards'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
