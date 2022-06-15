# Chapter 10 - Sequence modelling 

## Intro
* RNNs share paramaters across different parts of the model
* This makes it possible to apply the model to examples of different forms
* More information in Grave 2012

## Unfolding computational graphs
* Unfolding is unpacking recursively defined back to base case
* We can represent the unfolded recurrence as another function g^(t) of all the inputs
* But best to learn the transition function f which has the same inputs and parameters at each time step
* Computing the gradient of loss function is expensive because it has to run through all steps and cannot be parallelized because it has to be sequential

## Teacher forcing and networks with output recurrence