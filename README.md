# marchmathness
## What is it?
A model that predicts win probabilities between any two college basketball teams. 

The model is a neural network with one hidden layer, built using PyBrain and data supplied from [Kaggle's March Machine Learning Mania 2016 competition](https://www.kaggle.com/c/march-machine-learning-mania-2016) (but I finished too late and couldn't submit in time). The net trains on all college basketball regular season and tournament games since 2002, extracting features from [Ken Pomeroy's publicly available team statistics](http://kenpom.com/). Currently, it only uses adjusted team-centric ratings. 

### Bracketology
I first employed a greedy algorithm in which the team with the higher win probability is always picked to advance. 

In order to maximize points in a tournament challenge (using ESPN's scoring system of 320 possible points in each round), I also designed a dynamic programming algorithm that computes expected points for all possible scenarios and chooses the highest valued bracket.

#### Example
In 2016, both strategies ended up predicting the same Final Four (somewhat surprisingly all 1-seeds)
> (1) North Carolina, (1) Virginia, (1) Kansas, (1) Oregon

 and results
> (1) North Carolina > (1) Virginia

> (1) Kansas > (1) Oregon

> (1) Kansas > (1) North Carolina

but there were slight differences in the Elite Eight and further down.

The greedy alg had
> (1) North Carolina, (3) West Virginia, (1) Virginia, (3) Utah, (1) Kansas, (2) Villanova, (1) Oregon, (2) Oklahoma

The dynamic alg had
> (1) North Carolina, (6) Notre Dame, (1) Virginia, (6) Seton Hall, (1) Kansas, (6) Arizona, (1) Oregon, (2) Oklahoma

Note:

In the Round of 64, the greedy alg predicted 25/32 winners, and the dynamic alg predicted 20/32 winners.
In the Round of 32, the greedy alg predicted 9/16 winners, and the dynamic alg predicted 9/16 winners.