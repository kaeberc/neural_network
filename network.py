from input_layer import InputLayer
from inputs import InputNeuron
from layer import Layer
from typing import List
from clearable_layer import ClearableLayer
import numpy as np
from neuron import Neuron, ClearableNeuron
from modifiable_layers import ZERO_NODE

class NeuronNetwork:
  def __init__(self):
    self.__all_nodes:List[Neuron] = []
    self.__clearable_nodes:List[ClearableNeuron] = []
    self._input_nodes:List[InputNeuron] = []
    self._inputs = []
  def add_node(self,new_node):
    self.__all_nodes.append(new_node)
    if issubclass(type(new_node),ClearableNeuron):
      self.__clearable_nodes.append(new_node)
    if issubclass(type(new_node),InputNeuron):
      self._input_nodes.append(new_node)
  def __get_lambda(self):
    return lambda: self._inputs[len(self._inputs)]
  def add_input_node(self):
    self._inputs.append(0)
    self._input_nodes(InputNeuron(self.__get_lambda()))

class Network:
  def __init__(self, input_size:int):
    self.__first_layer:InputLayer = InputLayer(input_size)
    self._last_layer:Layer  = self.__first_layer
    self.__clearable_layers:List[ClearableLayer] = []
    self._input_size = input_size
  def add_layer(self, layer:Layer):
    self._last_layer = layer
    if issubclass(type(layer),ClearableLayer):
      self.__clearable_layers.append(layer)
  def output(self, inputs:List):
    if len(inputs)!= self._input_size:
      raise ValueError("Not the right number of inputs")
    self.__first_layer.input(inputs)
    ret_v = self._last_layer.output()
    self.__clear()
    return ret_v
  def __clear(self):
    for layer in self.__clearable_layers:
      layer.clear()
  def train(self, expected_input, expected_output,epochs,loss,loss_prime,print_output = True):
    if len(expected_input) != len(expected_output):
      raise ValueError("Input and output must be the same length")
    for i in range(epochs):
      err = 0
      for j in range(len(expected_input)):
        put_in = expected_input[j]
        self.__first_layer.input(put_in)
        output = self._last_layer.output()

        self.__clear()
        act_value = np.array(expected_output[j])
        pred_value = np.array(output)
        #print(act_value,pred_value)
        err += loss(act_value,pred_value)
        error = loss_prime(act_value,pred_value)

        self._last_layer.train(error)
      print('epoch %d/%d   error=%f' % (i+1, 1000, err))

