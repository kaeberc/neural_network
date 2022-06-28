from network import *
from complex_neurons import *
from losses import *
from activations import *

net = BrainNetwork()

x_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [[0],[1],[1],[0]]

#x_train = []

node1 = net.add_input_node()
node2 = net.add_input_node()

init_node = ModActivConnectNeuron([node1,node2],actv_func = tanh, actv_prime = tanh_prime,alpha=0.2)
sec_node = ModActivConnectNeuron([node1,node2],actv_func = tanh, actv_prime = tanh_prime,alpha=0.2)
#third_node = ModActivConnectNeuron([node1,node2],actv_func = tanh, actv_prime = tanh_prime)

out_node = ModActivConnectNeuron([init_node,sec_node],actv_func=tanh,actv_prime=tanh_prime,alpha=0.2)

#net.add_output_node(init_node)
net.add_node(init_node)
net.add_node(sec_node)
#net.add_node(third_node)
net.add_output_node(out_node)
continu= True
while continu:
  net.train(x_train,y_train,200,mse,mse_prime)
  for item in x_train:
    net.put_inputs(item)
    print(net.get_outputs())
    for item in net.get_outputs():
      bad = True
      if abs(item-0.5) > 0.3:
        bad = False
      if not bad:
        continu = False
