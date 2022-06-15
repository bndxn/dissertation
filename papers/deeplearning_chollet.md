# Chapter 6 - Deep learning for text and sequences

* Two main algorithms for sequence processing are recurrent neural networks and 1D convnets

## Text data and embeddings 

* Text is represented as a vector, with one-hot encoding for each value
* n-grams is a continuous sequence of words or characters
* breaking langugage into n-grams produces tokens, the process is tokenization
* one-hot encoding means creating a list of words, length m, and creating a vector for each word with 1 at the position of the word in the list, and zeros otherwise
* dense word vectors or embeddings - much less sparse, learned from data - also vectors but geometric location supposed to reflect semantic relationships
* embeddding layer is best understood as a dictionary that maps integer indices to dense vectors
* the embedding layer is trained each time

## Understanding RNNs

* RNN is a type of neural network that has an internal loop
* output_t = function(input_t , output_(t-1))


```
state_t = 0
for input_t in input_sequence:
    output_t = f(input_t, state_t) 
    state_t = output_t 
```

Adding in the activation function, and the transformation through dot with W and U 

```
state_t = 0 
for input_t in input_sequence:
    output_t = activation( dot(W, input_t) + dot(U, state_t) + b)
    state_t = output_t
```

An implementation in numpy:

```
import numpy as np

timesteps = 100
input_features = 32
output_feature = 64

inputs = np.random.random((timesteps, input_features)) # for this example, random values 

state_t = np.zeros((output_features)) # starts with an all-zero vector

W = np.random ((output_features, input_features)) # W is applied to the new inputs, of dim input_features
U = np.random ((output_features, output_features)) # U is applied to the previous state 
b = np.random ((output_features,))

successive_outputs = []
for input_t in inputs:
    output_t = np.tanh( np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t

final_output_sequence = np.concatenate(successive_outputs, axis=0)

```

* An RNN is a *for* loop that reuses quantities computed during the previous iteration of the loop
* Keras has a *SimpleRNN* layer which implements this, though it takes batches of sequences
* RNN layers in Keras can either return a single timestep or the whole sequence, using return_sequences=bool as argument
* If you stack many RNN layers together, the intermediate layers have to return full sequences of outputs

```
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
model = Sequential()
model.add(Embedding(10000,32))
model.add(SimpleRNN(32))
model.summary()
```
## 6.2 LSTM 

* SimpleRNN struggles to learn information seen many timesteps before, due to the vanishing gradient problem
* LSTM (Long Short-Term Memory) layer carries information across multiple timesteps
* Consider original RNN system, and add *carry track* 
* We do three transformations of the form `y= activation(dot(state_t, U)  + dot(input_t, W) + b)`
* We create three versions of `U` and `W`, indexed as `i,f,k`
* Then we create the carry state, `c_t+1 = i_t * k_t + c_t * f_t`
* Overall, the carry state equals the product of two activations **plus** the product of the previous state and one activation
* Activations `i_t` and `k_t` are about the present, `c_t` is about the past and is multiplied by `f_t` from the present
* These include a mixture of remembering and forgetting
* This deals with the vanishing gradients problem

