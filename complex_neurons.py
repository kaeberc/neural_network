from typing import Callable
from neuron import ClearableNeuron, Neuron
import numpy as np

def default(x):
  return x

def default_prime(x):
  return 1

class ActivationNeuron(Neuron):
  def __init__(self,inputs:Neuron, actv_func = default, actv_func_prime = default_prime):
    self.shape = 1
    self.prior_neuron = inputs
    self.actv = actv_func
    self.prime =actv_func_prime
    self.last_input = None
    self.error = None
  def output(self):
    self.last_input = self.prior_neuron.output()
    #print(self.actv(self.last_input))
    return self.actv(self.last_input)
  def backsend(self,error):
    self.error = error
    #print(error)
  def backtrain(self):
    self.prior_neuron.backsend(self.prime(self.last_input)*self.error)
  def prior(self):
    return [self.prior_neuron]

class ConnectedNeuron(ClearableNeuron):
  def __init__(self,inputs:list[Neuron], alpha:int=0.1, def_weights:np.array = None,must_clear = False):
    self.shape = len(inputs)
    self.alpha = alpha
    self.prior_neurons = inputs
    if def_weights:
      self.weights = def_weights
    else:
      self.weights = np.random.rand(self.shape)-0.5
      #self.weights = np.array([1.]*size)
    self.bias = np.random.rand()-0.5
    #self.bias = 1
    self.last_inputs = None
    self.cur_backwards = []
    self.cur_out = None
    self.__must_clear = must_clear

  def prior(self):
    return self.prior_neurons

  def output(self):
    if self.cur_out is None or self.__must_clear is False:
      self.cur_backwards = []
      ret_v = self.bias
      self.last_inputs=[]
      for weight,neuron in zip(self.weights,self.prior_neurons):
        val = neuron.output()
        self.last_inputs.append(val)
        #val*=weight
        ret_v += val*weight
      self.cur_out = ret_v
      return ret_v
    else:
      return self.cur_out

  def clear(self):
    self.cur_out = None
    self.cur_backwards = []

  def backsend(self,error):
    self.cur_backwards.append(error)

  def backtrain(self):
    assert self.cur_backwards != []
    error = np.sum(np.array(self.cur_backwards))

    for weight, neuron in zip(self.weights,self.prior_neurons):
      neuron.backsend(weight*error)
    self.weights -= error * self.alpha * np.array(self.last_inputs)
    self.bias -= error*self.alpha


class ModifiableConnectNeuron(ConnectedNeuron):
  def add_node(self, new_node):
    self.prior_neurons.append(new_node)
    self.weights = np.append(self.weights,np.random.rand()-0.5)
  def remove_node(self, node_to_remove:Neuron = None, index:int = None):
    if index is None and node_to_remove is None:
      raise ValueError("Needs a value to remove")
    if (not node_to_remove is None) and (not index is None):
      raise ValueError("Only provide one input")
    if not node_to_remove is None:
      index = self.prior_neurons.index(node_to_remove)

    self.prior_neurons.pop(index)
    self.weights.pop(index)

class ModActivConnectNeuron(ModifiableConnectNeuron):
  def __init__(self,inputs:list[Neuron],actv_func:Callable = default,actv_prime:Callable = default_prime, alpha:int=0.1, def_weights:np.array = None,must_clear = False):
    super().__init__(inputs, alpha, def_weights,must_clear)
    self.actv_func = actv_func
    self.actv_prime = actv_prime

  def output(self):
    self.__last_suboutput = super().output()
    return self.actv_func(self.__last_suboutput)

  def backsend(self,error):
    self.cur_backwards.append(self.actv_prime(self.__last_suboutput)*error)

  def backtrain(self):
    #assert self.cur_backwards != []

    if not self.cur_backwards == []:

      error = np.sum(np.array(self.cur_backwards))

      for weight, neuron in zip(self.weights,self.prior_neurons):
        neuron.backsend(weight*error)
      self.weights -= error * self.alpha * np.array(self.last_inputs)
      self.bias -= error*self.alpha