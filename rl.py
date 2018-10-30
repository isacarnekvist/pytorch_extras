import numpy as np
from torch.utils.data import Dataset


class ReplayBuffer(Dataset):
    """
    Usage:
    >>> buffer = ReplayBuffer(2)  # Creates a replay buffer with capacity 2
    >>> buffer.extend(['a', 'b'])
    >>> buffer[0]
    'a'
    >>> len(buffer)
    2

    Appending to a full buffer will drop random entries to
    keep same size:
    >>> buffer.append('c')
    >>> len(buffer)
    2

    Multiple rows can be appended using extend([...])
    >>> buffer.extend(['d', 'e', 'f'])

    The replay buffer is a child class of Dataset, so it can be used inside
    a dataloader:
    >>> from torch.utils.data import DataLoader
    >>> dataloader = DataLoader(buffer)
    >>> batch = next(iter(dataloader))
    """
    def __init__(self, capacity):
        self._contents = dict()
        self._len = 0
        self._capacity = capacity
        if capacity < 1:
            raise ValueError('capacity have to be > 0')
        if type(capacity) is not int:
            raise ValueError('')

    @property
    def capacity(self):
        return self._capacity

    def append(self, x):
        """
        Append one item to the buffer
        """
        if self._len == self._capacity:
            i = self._drop_random()
        else:
            i = self._len
            self._len += 1
        self._contents[i] = x

    def extend(self, iterable):
        """
        Extend buffer with multiple items contained in 'iterable'
        """
        for x in iterable:
            self.append(x)

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        return self._contents[i]

    def _drop_random(self):
        i = np.random.randint(0, len(self))
        del self._contents[i]
        return i
