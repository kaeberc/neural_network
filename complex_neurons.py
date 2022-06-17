from neuron import Neuron
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

class ConnectedNeuron(Neuron):
  def __init__(self,inputs:list[Neuron], alpha:int=0.1, def_weights:np.array = None):
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

  def prior(self):
    return self.prior_neurons

  def output(self):
    if self.cur_out is None:
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

  def backsend(self,error):
    self.cur_backwards.append(error)
    #print(error)

  def backtrain(self):
    assert self.cur_backwards != []
    error = np.sum(np.array(self.cur_backwards))
    self.cur_backwards = []

    for weight, neuron in zip(self.weights,self.prior_neurons):
      neuron.backsend(weight*error)
    #print(error)
    #print(self.weights)
    #print(self.last_inputs)
    #print(error * self.alpha * np.array(self.last_inputs))
    self.weights -= error * self.alpha * np.array(self.last_inputs)
    #print(self.weights)
    self.bias -= error*self.alpha
    #print(self.weights)
    #print()



