from abc import abstractclassmethod
from ast import Not


class Layer:
  @abstractclassmethod
  def train(self):
    raise NotImplementedError("Call to abstract class")
  @abstractclassmethod
  def get_nodes(self):
    raise NotImplementedError("Call to abstract class")
  @abstractclassmethod
  def train(self,error = None):
    raise NotImplementedError("Call to abstract class")
  @abstractclassmethod
  def output(self):
    raise NotImplementedError("Call to abstract class")