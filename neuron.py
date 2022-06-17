import numpy as np
import abc


class Neuron:
  @abc.abstractclassmethod
  def output(self):
    """
    Calculates current output
    """
    raise NotImplementedError("Call to abstract method")

  @abc.abstractclassmethod
  def backsend(self,error:np.array):
    """
    Calculates error and sends it backwards
    """
    raise NotImplementedError("Call to abstract method")

  def backtrain(self):
    """
    Updates current weights based on stored information
    """
    pass

  def prior(self):
    """
    Returns the previous neurons
    """
    return []