## Igor Gotlibovych from Octopus energy - [link](https://www.youtube.com/watch?v=p6mKFs6HVlg)
#  Deep Learning and Time Series Forecasting for Smarter Energy

## Context
- On generation side, prediction from renewables is less predictable
- Customers installing their own generation and using batteries
- Octopus has time series data of usage from smart meters
- 77K smart meters

## How Octopus approaches time series forecasting
- Lots of different approaches to time series forecasting, each from different communities
- TSF (Time series forecasting)
- Is it a regression problem? Yes, so can use scikit learn
- latent variables?
- octopus uses deep learning route as this scaled quite well
- they open sourced some code https://github.com/octoenergy/timeserio


## probability distributions
- aggregating across customers reduces spikes of usage at household level
- latent space groups customers
- errors are not good - not Gaussian
- why reduce MSE? Assumes underlying distribution is Gaussian, and we are fitting an estimator to the mean of the distribution
- Gaussian trick in Bishop: approximate any distribution by adding other models together
- generate sample one step at time reduces the variance
- tensorflow allows creation of arbitrary graph, can have quite complicated layers
- sharing the layers and graphs between models gives a lot of power

## summary
- time series can be seen as regression problem
- need to select features carefully
- yes deep learning but this does layer and parts of model sharing
