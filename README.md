# example

[![Build
Status](https://travis-ci.org/kazlewis/bmi203_finalproject.svg?branch=master)](https://travis-ci.org/kazlewis/bmi203_finalproject)

Implementation of three-layer feed forward neural network with backpropagation and error weighted derivaive updates. 
Includes bias nodes at both the input and hidden layer.

## usage

To use the package, first run

```
conda install --yes --file requirements.txt
```

to install all the dependencies in `requirements.txt`. Then the package's
main function (located in `example/__main__.py`) won't actually do anything when run (lol). 
Please edit run.py and run that file directly.


## testing

Testing is as simple as running

```
python -m pytest
```

from the root directory of this project.
