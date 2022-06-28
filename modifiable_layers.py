from clearable_layer import ClearableLayer
from layer import Layer
from overrider import overrides
from complex_neurons import ActivationNeuron, ModifiableConnectNeuron
from typing import Callable, List
from activation_layer import ActivationLayer
from inputs import InputNeuron

# Can't add functions like add_node since Connected only accepts connected nodes
class ModifiableLayer(Layer):
  pass

def zero():
  return 0

ZERO_NODE = InputNeuron(zero)


class ModifiableConnectedLayer(ClearableLayer,ModifiableLayer):
  def __init__(self, prev_layer:Layer, size:int, connection_map:dict = None, must_clear:bool = False):
    self._size = size
    self.__nodes:List[ModifiableConnectNeuron] = []
    prev_nodes = prev_layer.get_nodes()
    for i in range(size):
      if not connection_map is None:
        connections = []
        for index in connection_map[i]:
          connections.append(prev_nodes[index])
      else:
        connections = prev_nodes
      self.__nodes.append(ModifiableConnectNeuron(connections,must_clear = must_clear))

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

  def add_node(self,new_node):
    self.__nodes.append(new_node)

  def remove_node(self,node):
    indx = self.__nodes.index(node)
    if indx == -1:
      return False
    self.__nodes[indx] = ZERO_NODE
    return True


class ModifiableActivationLayer(ActivationLayer,ModifiableLayer):
  def add_node(self,new_node):
    self.__nodes.append(new_node)

  def remove_node(self,node):
    indx = self.__nodes.index(node)
    if indx == -1:
      return False
    self.__nodes[indx] = ZERO_NODE
    return True

"""
Contains both a modifiable layer and a clearable layer
"""
class ClusteredLayer(ModifiableLayer,ClearableLayer):
  def __init__(self,prev_layer:Layer, size:int,actv_func:Callable,actv_prime:Callable, connection_map:dict = None, must_clear = bool):
    self.prev_layer = prev_layer
    self.size = size
    self.must_clear = must_clear
    self.layer_1 = ModifiableConnectedLayer(prev_layer,size,connection_map,must_clear)
    self.layer_2 = ModifiableActivationLayer(self.layer_1,actv_func,actv_prime)
    self.__actv_func = actv_func
    self.__actv_prime = actv_prime
  def clear(self):
    self.layer_1.clear()

  @overrides(Layer)
  def output(self):
    return self.layer_2.output()

  @overrides(Layer)
  def train(self, error = None):
    self.layer_2.train(error)

  @overrides(Layer)
  def get_nodes(self):
    return self.layer_2.get_nodes()

  def get_weighted_nodes(self):
    return self.layer_1.get_nodes()

  def remove_node(self,node:ActivationNeuron):
    node_list = self.layer_2.get_nodes()
    self.layer_2.remove_node(node)

    # Technically serves no purpose
    indx = node_list.index(node)
    other_node = self.layer_1.get_nodes(other_node)
    self.layer_1.remove_node(other_node)

  def add_node(self,new_node:ModifiableConnectNeuron):
    layer2_node = ActivationNeuron(new_node,self.__actv_func,self.__actv_prime)
    self.layer_1.add_node(new_node)
    self.layer_2.add_node(layer2_node)




