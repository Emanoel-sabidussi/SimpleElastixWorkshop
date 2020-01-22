from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod

class BaseWeightedSequence(ABC):
    
    def __init__(self):
        super.__init__()

    @abstractmethod
    def forward_model(self):
        raise NotImplementedError("Forward_model not implemented")

    @abstractmethod
    def generate_weighted_sequence(self):
        raise NotImplementedError("Generate_weighted_sequence not implemented")

    @abstractmethod
    def gradients(self):
        raise NotImplementedError("Gradients not implemented")

        
    