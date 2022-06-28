from input_layer import InputLayer
from inputs import InputNeuron
from layer import Layer
from typing import List
from clearable_layer import ClearableLayer
import numpy as np
from neuron import Neuron, ClearableNeuron
from complex_neurons import ModActivConnectNeuron, ModifiableConnectNeuron
from modifiable_layers import ZERO_NODE
from overrider import overrides
import random as r
from activations import *

class NeuronNetwork:
  def __init__(self,needs_sort = True):
    self._all_nodes:List[Neuron] = []
    self.__clearable_nodes:List[ClearableNeuron] = []
    self._input_nodes:List[InputNeuron] = []
    self._inputs = []
    self.__output_nodes:List[Neuron] = []
    self.__input_count = 0
    self.needs_sort = needs_sort
  def add_node(self,new_node):
    self._all_nodes.append(new_node)
    if issubclass(type(new_node),ClearableNeuron):
      self.__clearable_nodes.append(new_node)
    if issubclass(type(new_node),InputNeuron):
      self._input_nodes.append(new_node)
  def __get_lambda(self,i):
    return lambda: self._inputs[i]
  def add_input_node(self) -> InputNeuron:
    self._inputs.append(0)
    self._input_nodes.append(InputNeuron(self.__get_lambda(self.__input_count)))
    self._all_nodes.append(self._input_nodes[-1])
    self.__input_count += 1
    return self._input_nodes[-1]
  def get_last_node(self):
    return self._all_nodes[-1]
  def add_output_node(self,node):
    if node in self._all_nodes:
      self.__output_nodes.append(node)
    else:
      self.add_node(node)
      self.__output_nodes.append(node)
  def get_outputs(self):
    ret_v = []
    for node in self.__output_nodes:
      ret_v.append(node.output())
    return ret_v
  def put_inputs(self,in_list):
    for i in range(len(in_list)):
      self._inputs[i] = in_list[i]
  def topo_sort(self):
    visited = {}
    cur_nodes: List[Neuron] = self._all_nodes.copy()
    self._all_nodes = []
    for node in cur_nodes:
      visited[node] = False
    for node in cur_nodes:
      self._topo_sort_util(node,visited)
  def _topo_sort_util(self,in_node:Neuron,visited):
    for node in in_node.prior():
      if visited[node] is False:
        self._topo_sort_util(node,visited)
    self._all_nodes.append(in_node)
    visited[in_node] = True
  def train_single(self, error:List):
    if len(error)!=len(self.__output_nodes):
      raise ValueError("Error does not match size")
    for i in range(len(error)):
      err = error[i]
      self.__output_nodes[i].backsend(err)
    if self.needs_sort:
      self.topo_sort()
    for node in reversed(self._all_nodes):
      node.backtrain()
  def __clear(self):
    for node in self.__clearable_nodes:
      node.clear()
  def train(self, expected_input, expected_output,epochs,loss,loss_prime,print_output = True):
    if len(expected_input) != len(expected_output):
      raise ValueError("Input and output must be the same length")
    for i in range(epochs):
      err = 0
      for j in range(len(expected_input)):
        put_in = expected_input[j]
        self.put_inputs(put_in)
        output = []
        if self.needs_sort:
          self.topo_sort()
        for node in self.__output_nodes:
          output.append(node.output())

        self.__clear()

        act_value = np.array(expected_output[j])
        pred_value = np.array(output)
        #print(pred_value)
        #print(act_value,pred_value)
        err += loss(act_value,pred_value)
        error = loss_prime(act_value,pred_value)
        self.train_single(error)
        #self._all_nodes[-1].backsend(error)
        #for node in reversed(self._all_nodes):
          #node.backtrain()
      print('epoch %d/%d   error=%f   nodes = %d' % (i+1, epochs, err,len(self._all_nodes)))





class ModifiableNeuronNetwork(NeuronNetwork):
  @overrides(NeuronNetwork)
  def __init__(self,needs_sort = True):
    super().__init__(needs_sort)
    self._modifiable_neurons:List[ModifiableConnectNeuron] = []

  @overrides(NeuronNetwork)
  def add_node(self, new_node):
    super().add_node(new_node)
    if issubclass(type(new_node),ModifiableConnectNeuron):
      self._modifiable_neurons.append(new_node)

  @overrides(NeuronNetwork)
  def topo_sort(self):
    self.__mod_copy = self._modifiable_neurons.copy()
    self._modifiable_neurons = []
    super().topo_sort()

  @overrides(NeuronNetwork)
  def _topo_sort_util(self,in_node:Neuron,visited):
    for node in in_node.prior():
      if visited[node] is False:
        self._topo_sort_util(node,visited)
    self._all_nodes.append(in_node)
    if in_node in self.__mod_copy:
      self._modifiable_neurons.append(in_node)
    visited[in_node] = True

class SynaptoNetwork(ModifiableNeuronNetwork):
  @overrides(ModifiableNeuronNetwork)
  def __init__(self,needs_clear = True):
    super().__init__(needs_clear)
    self._modifiable_neurons_max = {}
    self._modifiable_neurons_min = {}
  @overrides(NeuronNetwork)
  def train_single(self,error):
    # Do normal training
    super().train_single(error)
    self.synaptogenesis()

  def synaptogenesis(self):
    # For every node we can change the inputs for
    for node in self._modifiable_neurons:
      err = sum(node.cur_backwards)
      # Check that this is a new input and record the new input
      if not node in self._modifiable_neurons_max:
        self._modifiable_neurons_min[node] = 0
        self._modifiable_neurons_max[node] = 0
        continue

      if err > self._modifiable_neurons_max[node]:
        self._modifiable_neurons_max[node] = err
      elif err < self._modifiable_neurons_min[node]:
        self._modifiable_neurons_min[node] = err
      else:
        # If not a new situation, continue through the list
        continue

      node_index = self._all_nodes.index(node)
      priors = node.prior()
      # For all nodes before this one
      for i in range(node_index):
        cur_node:Neuron = self._all_nodes[i]
        # If node is 'active'
        if abs(cur_node.output()) > 0.5:
          if not cur_node in priors:
            node.add_node(cur_node)

class NGNetwork(ModifiableNeuronNetwork):

  @overrides(NeuronNetwork)
  def train_single(self,error):
    super().train_single(error)
    self.neurogenesis()

  def neurogenesis(self):
    #k = 0
    for node in self._modifiable_neurons:
      #k+=1
      #print(k)
      node_index = self._all_nodes.index(node)
      err = sum(node.cur_backwards)

      if abs(err) > 0.5 and len(node.prior()) < 5: #and r.random() > 0.9:
        priors = [self._all_nodes[node_index-1]]

        for i in range(2):
          if (node_index-1-i) >= 0:
            priors.append(self._all_nodes[node_index-1-i])
        new_node = ModActivConnectNeuron(priors,actv_func = tanh, actv_prime = tanh_prime,alpha = node.alpha)
        node.add_node(new_node)
        self.add_node(new_node)

class BrainNetwork(NGNetwork,SynaptoNetwork):
  def __init__(self,needs_clear = True, develop_interval = 100):
    super().__init__()
    self.__count = 0
    self.__interval = 100

  @overrides(NGNetwork)
  @overrides(SynaptoNetwork)
  def train_single(self,error):
    ModifiableNeuronNetwork.train_single(self,error)
    if self.__count == self.__interval:
      print("Test")
      self.synaptogenesis()
      self.neurogenesis()
      self.__count = 0
    self.__count += 1




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
      print('epoch %d/%d   error=%f' % (i+1, epochs, err))

