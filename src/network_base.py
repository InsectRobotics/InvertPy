from utils import RNG
import numpy as np


class Network(object):

    def __init__(self, rng=RNG, dtype=np.float32):
        """
        :param gain:
        :type gain: float, int
        :param rng: the random state generator
        :type rng: np.random.RandomState
        :param dtype: the type of the values in the network
        :type dtype: Type[np.dtype]
        """
        self.dtype = dtype
        self.rng = rng

        self.params = []

        self.__update = False

    @property
    def update(self):
        return self.__update

    @update.setter
    def update(self, value):
        self.__update = value

    def reset(self):
        self.__update = False
