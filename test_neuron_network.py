from network import NeuronNetwork
from complex_neurons import ConnectedNeuron

net = NeuronNetwork()

node1 = net.add_input_node()
node2 = net.add_input_node()
node3 = net.add_input_node()
net.add_output_node(node1)
net.add_output_node(node2)
obsv_node = ConnectedNeuron([node1,node2,node3])
net.add_node(obsv_node)
net.add_output_node(obsv_node)
net.add_output_node(node3)
net.put_inputs([1,2,3])
net.topo_sort()
print(net.get_outputs())
