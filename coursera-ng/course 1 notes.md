# Week 1

Structure
1. Course starts with NNs and deep learning 
2. Improving DNNs: hyperparameters, regularization, optimization
3. Structuring your ML project
4. Convolutional neural networks - often applied to images
5. Sequence models - e.g. RNNs, LSTM (long short term memory problems)

## What is a neural network? - lecture

* Predicting the price of houses
* Linear regression has input/output mapping
* ReLU is rectified - which means taking a max with zero
* Inputs variables maybe tell you a feature, e.g. houses imply the "family size" and the "walkability"
* x is four inputs, y is the price, and the middle neurons 
* the hidden units in a neural network can take **all** input features - the input and middle layer are **densely connected**

## Supervised learning - lecture

* Big gains in several areas where there is a clear input and output
* Value has been from cleverly selecting what should be x and y 
* For image applications, often use convolutional neural networks
* Audio represented as 1-D time series, which often uses a RNN, recurrent neural network
* Language also done using RNN as a sequence
* Driving and radars are more mixed and custom
* Representations of CNNs are as blocks and cubes
* Unstructured data: includes audio, image, text
* Computers now much better at dealing with unstructured data
* Basic ideas behind NNs have been around for decades

## Why is deep learning now taking off?

* Scale drives deep learning - more data makes it easier to train
* Large neural networks work best with large data
* To get the best performance, we also want bigger models
* Reliable methods are to get a bigger model or add more data to it
* m denotes number of training examples
* there has also been algorithmic innovations, about making NNs run faster 
* example of algorithmic innovation: switching from sigmoid from ReLU, which makes gradient descent work much faster
* shorter cycle means easier to get new results faster, try more ideas
* faster computation speeds up experimental rate, helps practitioners and researchers 

# Week 2

## Notation
* We often want to process m examples as part of training, but we can avoid having to use a *for loop*  for each instance
* In NNs, we often have a forward pass and backward pass - we learn why these two passes in different directions
* We examine these using logistic regression
* Binary classification - recognising an image as either a cat or not a cat, y denotes output label
* RGB images stored as three matrices - if it 64x64 image, then 3 matrices size 64x64
* We convert this to one long vector - 3*64*64 = 12,288 long
* n_x represents dimensions of input feature vector
* Training example: (x, y), where x is in R^{n_x}, y in {0,1}
* m training examples
* big matrix X has m columns, one for each of the m inputs
* Y is matrix 1xm

## Logistic regression - model

* Given x, we want y_hat = P(y=1|x)
* One approach y_hat = wTx + b, but we want y to be between 0 and 1 but this
* We scale it: y_hat = sigmoid(wTx + b)

## Logistic regression - cost function

* We want to learn parameters w and b for y_hat = sigmoid(wTx +b)
* We define loss function, square error is good but it makes gradient descent not work well
* Loss function for classification can use logs which covers the true 1/0 and output 1/0 cases well
* Cost function is for the entire training set: 1/m of the sum of the loss on each training examples

## Gradient descent

* We want to find w and b such that the cost function is minimised
* The cost function is some surface, so the height of the surface at the point is the cost function for that input
* The cost function given above is convex, which means gradient descent will converge on a globally optimal solution
* Gradient descent takes a step in the steepest downhill direction
* w := w - alpha * gradient of cost function w.r.t w
* w := w - alpha * dw 

## Derivatives

* Nudging the inputs leads to changes in the outputs
* Derivative means slope, the height divided by the width of the triangle of changing the inputs and seeing how the outputs change
 
## More derivative examples

* On non-linear functions, the slope will change on different parts of the function
* Derivate of log(a) is l/a

## Computation graph

* Computations made by forward pass: ouptut of NN, followed by backward pass: get gradients and or compute derivatives
* Computation graph shows why
* How to calculate J(a,b,c) = 3(a + bc)? Could start with u = b*c, then v = a + u, then j = 3v, do each step
* Computation graph shows how inputs combine to give intermediate results u and v


# Week 1 - Vectorization 

* Write operations as vectors where possible, avoid for-loops
* Writing multiplications or other element-wise operations as vectors allows them to be processed in parallel which speeds up operations
* SIMD (single instruction, multiple data) is a type of parallel processing 
* For example, np.dot(a,b) much faster than looping through a and b to multiply elements
* Another example: multiplying matrix A (m by n) by array v (1 by n), np.dot(A, v) will be much faster than for i, for j A[i][j] * v[j]
* Another example, np.exp(v) will return an array of the same size as v, but where each item is an exponent
* Another one: np.log(v), np.abs(a), np.max(v,0) does max of each element and 0, v**2 does element-wise square, 1/v does element-wise division

## Implementing vectorisation in logistic regression

* Replace dw1 and dw2 with dw = np.zeros((n-x, 1))
* Replace dw with dw += x^(i)dz^(i)

(Skipping this)

# Week 3

# Shallow neural networks 
