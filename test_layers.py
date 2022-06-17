from input_layer import InputLayer
from losses import *
from input_layer import InputLayer
from fully_connected_layer import FCLayer
from activation_layer import ActivationLayer
from activations import *

# Training Data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])


# Set up network
layer0 = InputLayer(2)
layer1 = FCLayer(layer0,3)
layer2 = ActivationLayer(layer1,tanh,tanh_prime)
layer3 = FCLayer(layer2,1)
layer4 = ActivationLayer(layer3,tanh,tanh_prime)

# Set up loss functions
loss = mse
loss_prime = mse_prime

for i in range(1000):
  err = 0
  for j in range(len(x_train)): #
    put_in = x_train[j][0]
    print(put_in)
    layer0.input(put_in)
    output = layer4.output()

    # Clear layers with activation functions
    layer1.clear()
    layer3.clear()

    err += loss(y_train[j],output)
    error = loss_prime(y_train[j],output)

    layer4.train(error)

  print('epoch %d/%d   error=%f' % (i+1, 1000, err))


  err /= len(x_train)
