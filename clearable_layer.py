from torch import InterfaceType
from layer import Layer

class ClearableLayer(Layer):
  def clear(self):
    raise NotImplementedError("Call to abstract class")
