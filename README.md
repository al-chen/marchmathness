# marchmathness
## What is it?
A model that predicts win probabilities between any two college basketball teams. 

The model is a neural network with one hidden layer, built using PyBrain and data supplied from [Kaggle's March Machine Learning Mania 2016 competition](https://www.kaggle.com/c/march-machine-learning-mania-2016) (but I finished too late and couldn't submit in time). The net trains on all college basketball regular season and tournament games since 2002, extracting features from [Ken Pomeroy's publicly available team statistics](http://kenpom.com/). Currently, it only uses adjusted team-centric ratings. 

### Bracket Strategy
I first employed a greedy algorithm in which the team with the higher win probability is always picked to advance. 

In order to maximize points in a tournament challenge (using ESPN's scoring system of 320 possible points in each round), I also designed a recursive algorithm with memoization that calculates expected points chooses the highest valued bracket. In order to save computations, if a team is favored to win by at least a certain cutoff threshold (default 90%), then I automatically advance that team and do not consider the bracket in which that team loses. The assumption is that it is severely unlikely for a bracket that has a big favorite losing to have more value than the bracket with the favorite winning.

In practice, these algorithms unsurprisingly produced virtually identical brackets, as the brackets that advance the teams that are most likely to win also generally score higher.

---

## 2024-2025: Modern Pipeline

Starting in 2024, the project was rebuilt using TensorFlow/Keras with a dual-model approach:

- **WIN model**: Binary classification (win/loss) — outputs win probabilities directly
- **MOV model**: Regression on margin of victory — probabilities derived from predicted spreads

Both models are feed-forward neural networks trained on historical regular season and tournament data, using KenPom adjusted efficiency metrics as features. The final submission blends predictions from both models.

### Directory Structure

```
marchmathness/
├── archive/          # Legacy 2016 PyBrain code (originally src/)
├── kenpom/           # Shared KenPom data across years
├── 2024/             # 2024 competition workspace
│   ├── march-madness-2024.ipynb
│   └── 2024_win_probability_matrix.json
├── 2025/             # 2025 competition workspace
│   ├── marchmathness.ipynb          # Main production notebook
│   ├── derive.ipynb                 # Feature engineering experiments
│   ├── derivev2.ipynb               # Feature engineering v2
│   ├── 2025_mov_win_probability_matrix.json
│   └── 2025_win_win_probability_matrix.json
├── march-machine-learning-mania-2016-v2/   # 2016 Kaggle data
└── readable_output_2016.csv
```

Each year directory also contains a `march-machine-learning-mania-YYYY/` folder with Kaggle competition data. These are gitignored due to size (~100-270MB) and should be downloaded from Kaggle.

### Adding a New Year

1. Copy the previous year's main notebook into a new `YYYY/` directory
2. Update the `SEASON` variable in the notebook
3. Download the new Kaggle competition data into `YYYY/march-machine-learning-mania-YYYY/`
4. Refresh KenPom data in `kenpom/`
5. Run the notebook and export predictions

### Dependencies

- tensorflow
- pandas
- numpy
- scikit-learn
- beautifulsoup4
- tqdm