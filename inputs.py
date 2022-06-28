from neuron import Neuron
import numpy as np

def default():
  return 1

class InputNeuron(Neuron):
  def __init__(self, obsv_func=default):
    self.obsv_func = obsv_func

  def output(self):
    return self.obsv_func()

  def backsend(self,error):
    pass

  def backtrain(self):
    pass
  def prior(self):
    return []
