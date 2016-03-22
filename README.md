# marchmathness
## What is it?
A model that predicts win probabilities between any two college basketball teams. 

The model is a neural network with one hidden layer, built using PyBrain and data supplied from [Kaggle's March Machine Learning Mania 2016 competition](https://www.kaggle.com/c/march-machine-learning-mania-2016) (but I finished too late and couldn't submit in time). The net trains on all college basketball regular season and tournament games since 2002, extracting features from [Ken Pomeroy's publicly available team statistics](http://kenpom.com/). For now, it uses only adjusted team-centric ratings. 

For filling out a tournament bracket, I employed a greedy algorithm in which the team with the higher win probability is always picked to advance. In order to maximize points in a tournament challenge, I would need to design around the specific scoring system and compute expected points for all possible scenarios. 