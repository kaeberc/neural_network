from complex_neurons import *
from inputs import *
from losses import *
from activations import *


def get_x():
  global x
  return x

def get_y():
  global y
  return y

def get_actual():
  global x, y
  return x+y

"""
x = 1
y = 3

r1 = InputNeuron(get_x)
r2 = InputNeuron(get_y)
test_inputs = [r1,r2]
test = ConnectedNeuron(2,test_inputs,alpha=0.01)


for x in range(100):
  for y in range(10):
    cur_est = test.output()
    loss =  get_actual()-cur_est
    print(cur_est,get_actual(),loss)
    test.backsend(loss)
    test.backtrain()
"""


x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

input_x = InputNeuron(get_x)
input_y = InputNeuron(get_y)

layer1_1 = ModActivConnectNeuron([input_x,input_y],actv_func=tanh,actv_prime = tanh_prime,alpha=0.2)
layer1_2 = ModActivConnectNeuron([input_x,input_y],actv_func=tanh,actv_prime = tanh_prime,alpha=0.2)
layer1_3 = ModActivConnectNeuron([input_x,input_y],actv_func=tanh,actv_prime = tanh_prime,alpha=0.2)

#layer2_1 = ActivationNeuron(layer1_1,tanh,tanh_prime)
#layer2_2 = ActivationNeuron(layer1_2,tanh,tanh_prime)
#layer2_3 = ActivationNeuron(layer1_3,tanh,tanh_prime)

#layer3_1 = ConnectedNeuron([layer1_1,layer1_2,layer1_3],0.2)

#layer4_1 = ActivationNeuron(layer3_1,tanh,tanh_prime)

layer2_1 = ModActivConnectNeuron([layer1_1,layer1_2,layer1_3],actv_func = tanh, actv_prime = tanh_prime, alpha=0.2)

loss = mse
loss_prime = mse_prime
for i in range(1000):
  err = 0


  for j in range(len(x_train)):
    #print(j)
    output = x_train[j][0]
    x = output[0]
    y = output[1]

    #output = layer4_1.output()
    output = layer2_1.output()

    err += loss(y_train[j],output)

    error = loss_prime(y_train[j],output)
    #print(output,error)


    # Do this better
    layer2_1.backsend(error)
    layer2_1.backtrain()
    #print(layer3_1.weights)
    #print(layer3_1.cur_backwards)
    #layer3_1.backtrain()
    #print()

    #layer2_1.backtrain()
    #layer2_2.backtrain()
    #layer2_3.backtrain()

    layer1_1.backtrain()
    layer1_2.backtrain()
    layer1_3.backtrain()

    layer1_1.clear()
    layer1_2.clear()
    layer1_3.clear()
    layer2_1.clear()
    #print(error[0][0],layer3_1.weights,layer3_1.bias)
    #print(layer1_1.weights,layer1_2.weights)

  err /= len(x_train)

  print('epoch %d/%d   error=%f' % (i+1, 1000, err))

# Test output

samples = len(x_train)
result = []
compare_results = []

for i in range(samples):
  output = x_train[i]
  x = output[0][0]
  y = output[0][1]
  compare_results.append((x,y))
  result.append(layer2_1.output())
  layer1_1.clear()
  layer1_2.clear()
  layer1_3.clear()
  layer2_1.clear()

print(compare_results)
print(result)
print(layer2_1.weights)
print(layer1_1.weights,layer1_2.weights)
print(tanh_prime(1))