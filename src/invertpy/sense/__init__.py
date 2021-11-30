"""
Python package that implements sensors of the invertebrates. The sensors are the only way that we can
experience our environment. Depending on the available sensors, we can perceive different aspects of it.
Invertebrates have a set of very different sensors from mammals (humans). In this package we try to capture
what invertebrates sense and how do they transform their surrounding environment into neural responses
that are fed to their brain.
"""

from .vision import CompoundEye
from .polarisation import PolarisationSensor
from .olfaction import Antennas
from .sensor import Sensor
