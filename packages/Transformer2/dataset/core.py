import os
import tensorflow_datasets as tfds
from .base import Base
from .newsroom import Newsroom
from .ptToEnTranslate import PtToEnTranslate

class Core:

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.dataSetProcessor = Base(self.path)
        return
    
    def get(self):
        if (self.name == 'newsroom'):
            self.dataSetProcessor = Newsroom(self.path)
        if (self.name == 'pt_to_en_translate'):
            self.dataSetProcessor = PtToEnTranslate(self.path)

        return self.dataSetProcessor
    
    
   