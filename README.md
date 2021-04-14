# InvertPy ![GitHub top language](https://img.shields.io/github/languages/top/evgkanias/InvertBrain) [![GitHub license](https://img.shields.io/github/license/evgkanias/InvertBrain)](https://github.com/evgkanias/InvertBrain/blob/main/LICENSE) [![Build Status](https://travis-ci.com/evgkanias/InvertBrain.svg?token=tyo7V4GZ2Vq6iYPrXVLD&branch=main)](https://travis-ci.com/evgkanias/InvertBrain)

This Python package aims to collect and implement computational models that are used
in the literature and are relevant to the invertebrate processing, from the environment
to sensor responses and to deeper neural responses in the invertebrate brain. It is
split in 3 sub-packages: the **InvertBrain**, the **InvertSense** and the **InvertIO**;
which are responsible for brain, sensor and data processing respectively.

[InvertSy](https://github.com/evgkanias/InvertSimulations) is a separate package that
implements environments such as the *sky* and an *AntWorld of vegetation*, using
simple-to-install python packages, e.g. NumPy and SciPy. These environments contain
information that humans can or cannot detect but invertebrates definitely can (e.g.
polarised light in the sky). This package also contains some examples of how to use
the **InvertPy** package.



## InvertBrain

Python package that embodies brain computations of the invertebrates.
This package contains python implementations of computational models for components
and functions in the insect brain. The brain components include the mushroom bodies
(MB) and central complex (CX). A variety of customised functions are also available
allowing to change their characteristics and create your own brain models.

## InvertSense

Python package that implements sensors of the invertebrates.
The sensors are the only way that we can experience our environment. Depending on the
available sensors, we can perceive different aspects of it. Invertebrates have a set
of very different sensors from mammals (humans). In this package we try to capture
what invertebrates sense and how do they transform their surrounding environment into
neural responses that are feed to their brain.

## InvertIO

Python package that allows saving and loading brain and sensor parameters to files.

## Environment

In order to be able to use this code, the required packages are listed below:
* [Python 3.8](https://www.python.org/downloads/release/python-380/)
* [NumPy](https://numpy.org/)  >= 1.20.1
* [SciPy](https://www.scipy.org/) >= 1.6.0
* [scikit-learn](https://scikit-learn.org/stable/) >= 0.23.2

## Author

The code is written by [Evripidis Gkanias](https://evgkanias.github.io/).

## Copyright

Copyright &copy; 2021, Insect robotics Group, Institute of Perception,
Action and Behaviour, School of Informatics, the University of Edinburgh.
