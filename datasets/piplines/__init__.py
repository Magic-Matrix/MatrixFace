
class BasePipeline(object):
    def __init__(self) -> None:
        pass
    def __call__(self, results:dict) -> dict:
        return results
    

from .transforms import *
from .utils import *
from .targets import *