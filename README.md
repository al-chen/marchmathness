# marchmathness
## What is it?
A model that predicts win probabilities between any two college basketball teams. 

The model is a neural network with one hidden layer, built using PyBrain and data supplied from [Kaggle's March Machine Learning Mania 2016 competition](https://www.kaggle.com/c/march-machine-learning-mania-2016) (but I finished too late and couldn't submit in time). The net trains on all college basketball regular season and tournament games since 2002, extracting features from [Ken Pomeroy's publicly available team statistics](http://kenpom.com/). Currently, it only uses adjusted team-centric ratings. 

### Bracket Strategy
I first employed a greedy algorithm in which the team with the higher win probability is always picked to advance. 

In order to maximize points in a tournament challenge (using ESPN's scoring system of 320 possible points in each round), I also designed a recursive algorithm with memoization that calculates expected points chooses the highest valued bracket. In order to save computations, if a team is favored to win by at least a certain cutoff threshold (default 90%), then I automatically advance that team and do not consider the bracket in which that team loses. The assumption is that it is severely unlikely for a bracket that has a big favorite losing to have more value than the bracket with the favorite winning.

In practice, these algorithms unsurprisingly produced virtually identical brackets, as the brackets that advance the teams that are most likely to win also generally score higher. 