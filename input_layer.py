from layer import Layer
from inputs import InputNeuron
from typing import List
from collections.abc import Callable
from overrider import overrides

class InputLayer(Layer):
  def __init__(self,size, input_funcs: List[Callable] = None):
    self._size = size
    self.__nodes:List[InputNeuron] = []
    self.__inputs = [0]*size
    if input_funcs is None:
      for i in range(size):
        cur_func = self.__get_lambda(i)
        self.__nodes.append(InputNeuron(cur_func))
    else:
      assert len(input_funcs) >= size
      for i in range(size):
        cur_func = self.__get_lambda(i)
        self.__nodes.append(InputNeuron(input_funcs[i]))
  def __get_lambda(self, i):
    return lambda: self.inputs[i]

  def input(self,new_in:List):
    self.inputs = new_in

  @overrides(Layer)
  def output(self):
    out_arr = []
    for i in range(self._size):
      out_arr.append(self.__nodes[i].output())
    return out_arr
  @overrides(Layer)
  def get_nodes(self, error = None):
    return self.__nodes.copy()

  @overrides(Layer)
  def train(self):
    pass
