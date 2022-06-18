from layer import Layer
from complex_neurons import ConnectedNeuron
from typing import List
from overrider import overrides
from clearable_layer import ClearableLayer

class FCLayer(ClearableLayer):
  def __init__(self, prev_layer:Layer, size, must_clear = False):
    self._size = size
    self.__nodes:List[ConnectedNeuron] = []
    prev_nodes = prev_layer.get_nodes()
    for i in range(size):
      self.__nodes.append(ConnectedNeuron(prev_nodes,must_clear = must_clear))

    self.prev_layer = prev_layer
    self.__must_clear = must_clear

  def clear(self):
    if self.__must_clear is True:
      for node in self.__nodes:
        node.clear()

  @overrides(Layer)
  def output(self):
    out_arr = []
    for i in range(self._size):
      out_arr.append(self.__nodes[i].output())
    return out_arr

  @overrides(Layer)
  def train(self, error = None):
    if error is None:
      for node in self.__nodes:
        node.backtrain()
    else:
      for element, node in zip(error,self.__nodes):
        node.backsend(element)
        node.backtrain()
    self.prev_layer.train()

  @overrides(Layer)
  def get_nodes(self):
    return self.__nodes.copy()