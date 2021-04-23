"""
Python package that embodies brain computations of the invertebrates. This package contains python implementations
of computational models for components and functions in the insect brain. The brain components include the
mushroom bodies (MB) and central complex (CX). A variety of customised functions are also available allowing to
change their characteristics and create your own brain models.
"""

from .mushroombody import MushroomBody, WillshawNetwork
from .centralcomplex import CentralComplex
from .compass import CelestialCompass, PolarisationCompass, SolarCompass
from .component import Component
from .plasticity import get_learning_rule
