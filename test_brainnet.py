from network import *
from complex_neurons import *
from losses import *
from activations import *

net = NGNetwork()

x_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [[0],[1],[1],[0]]

node1 = net.add_input_node()
node2 = net.add_input_node()

init_node = ModActivConnectNeuron([node1,node2],actv_func = tanh, actv_prime = tanh_prime)
sec_node = ModActivConnectNeuron([node1,node2],actv_func = tanh, actv_prime = tanh_prime)
third_node = ModActivConnectNeuron([node1,node2],actv_func = tanh, actv_prime = tanh_prime)

out_node = ModActivConnectNeuron([init_node,sec_node,third_node],actv_func=tanh,actv_prime=tanh_prime)

#net.add_output_node(init_node)
net.add_node(init_node)
net.add_node(sec_node)
net.add_node(third_node)
net.add_output_node(out_node)

net.train(x_train,y_train,5000,mse,mse_prime)
for item in x_train:
  net.put_inputs(item)
  print(net.get_outputs())
