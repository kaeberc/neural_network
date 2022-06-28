from activation_layer import ActivationLayer
from network import Network
from fully_connected_layer import FCLayer
from losses import mse, mse_prime
from activations import *

x_train = [[0,0],[0,1],[1,0],[1,1]]
y_train = [[0],[1],[1],[0]]

net = Network(2)
net.add_layer(FCLayer(net._last_layer,3))
net.add_layer(ActivationLayer(net._last_layer,tanh,tanh_prime))
net.add_layer(FCLayer(net._last_layer,1))
net.add_layer(ActivationLayer(net._last_layer,tanh,tanh_prime))

net.train(x_train,y_train,500,mse,mse_prime)