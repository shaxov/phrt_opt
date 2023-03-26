from . import ops
from . import eig
from . import parsers
from . import methods
from . import metrics
from . import callbacks
from . import strategies
from . import linesearch
from . import initializers
from collections import namedtuple
from ._contrib import retrieve_transmission_matrix, compute_camera_bias
from .typedef import *


Optimizer = namedtuple('Optimizer', ['name', 'callable', 'params'])
Initializer = namedtuple('Initializer', ['name', 'callable'])
