from . import ops
from . import methods
from . import metrics
from . import callbacks
from . import strategy
from . import linesearch
from . import initializers
from collections import namedtuple
from .typedef import *


Optimizer = namedtuple('Optimizer', ['name', 'method', 'params'])
Initializer = namedtuple('Initializer', ['name', 'method', 'params'])
