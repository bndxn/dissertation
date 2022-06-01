# WeatherBench Probability: A benchmark dataset for probabilistic medium-range weather forecasting along with deep learning baseline models

[Arxiv](https://arxiv.org/abs/2002.00469)

From Stephen Rasp's Twitter:

We tested three method of creating probabilistic ML predictions:
a) Mote-Carlo dropout
b) Predicting a parametric distribution
c) Predicting a discretized distribution (a la MetNet)
We find that b) and c) work quite well but dropout severely underestimates spread. 

# Paper

* Weather predictions should be probabilistic
* Previous approaches to generating probabilstic distributions: random initial condition perturbations, singular vector initial condition perturbations, and random seeds for the neural networks
* Target variables: 500 hPa geopotential, 860 hPa temperature, 2m temperature, and 6h accumulated precipitation
* They use a deep Resnet, and try MC dropout, parametric prediction (draw the parameters from a distribution), and categorical prediction with different bins
* Authors also say Gaussian is poor fit for precipitation - but maybe not so for PV yield
* Idea : could extend this categorical prediction
* Grid size : 5.625 degrees, seen as quite coarse
* Discussion on parametric and categorical approaches: parametric works well when distribution is known, 
