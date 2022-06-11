# Sequence models

## Why use sequence models?
* Includes RNNs
* Speech recognition is a sequence of audio and the output is also audio over time
* DNA sequence analysis
* All examples all supervised data

## Notation 

* Name-entity recognition
* x is input, sequence of 9 words
* y is 1 or 0 for whether it's a name
* x^<1> represents the first item
* T_x = 9 means input has 9 items
* X^(i) denotes the ith training example
* t is the element, i is which training example
* use a vocabulary, i.e. a dictionary, e.g. 10,000 words
* can use one-hot representation (vector of 10,000 words with 1 for the word, 0s otherwise)
* word not in vocabulary is < unk >

## Recurrent neural network model 

* Could try a normal NNs:
* Problems: inputs and outputs can be different lengths in different examples, but also it doesn't share features learned across **different positions** in the text
* You want things learned in one place to generalise in other places too
* RNN: take first item, predict output y^, then move to the second word and put this in, but also inputs activation value from first time step
* At each timestep, the activation from the previous timestep is included
* For timestep t0, often 0 is input
* On graphical representations, people sometimes use a loop
* RNNs sometimes only take inputs from earlier in the sequence (unidirectional RNNs)
* People usually use tanh, but sometimes use ReLU

## Backpropagation in a RNN

*  