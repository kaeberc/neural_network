from layer import Layer
from complex_neurons import ActivationNeuron
from typing import List
from collections.abc import Callable
from overrider import overrides

class ActivationLayer(Layer):
  def __init__(self, prev_layer:Layer, activation_function:Callable, activation_prime:Callable):
    if activation_function is None or activation_prime is None:
      raise ValueError("Requires an activation function and its prime")

    self.prev_layer = prev_layer
    self._size = len(self.prev_layer.get_nodes())
    self.__nodes: List[ActivationNeuron] = []
    for node in self.prev_layer.get_nodes():
      self.__nodes.append(ActivationNeuron(node,activation_function,activation_prime))

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

