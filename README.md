# College Football Game Predictions

machine learning that predicts the outcome of any Division I college football game. Data are from 2008 - 2021 seasons

## Installation
```bash
conda env create -f cfb_env.yaml
```

## Usage

```python
python cfb_ml.py
```
### Current prediction accuracies
```bash
Removed features: ['penalty_yards']

# Best hyperparameters
GradientBoostingClassifier - best params:  {'criterion': 'friedman_mse', 'learning_rate': 0.30000000000000004, 'loss': 'log_loss', 'max_depth': 2, 'max_features': 'log2', 'n_estimators': 300}
RandomForestClassifier - best params:  {'criterion': 'gini', 'max_depth': 4, 'max_features': 'log2', 'n_estimators': 300}
DecisionTreeClassifier - best params:  {'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'splitter': 'best'}
AdaClassifier - best params:  {'algorithm': 'SAMME.R', 'learning_rate': 1.0, 'n_estimators': 150}
LogisticRegression - best params: {'C': 1.5, 'max_iter': 800, 'penalty': 'l2', 'solver': 'lbfgs'}
MLPClassifier - best params:  {'learning_rate': 'invscaling', 'learning_rate_init': 0.004, 'max_iter': 700, 'solver': 'lbfgs'}
KNeighborsClassifier - best params:  {'algorithm': 'auto', 'n_neighbors': 100, 'p': 1, 'weights': 'distance'}

GradientBoostingClassifier accuracy 0.8107835726883346
RandomForestClassifier accuracy 0.7727387251196774
DecisionTreeClassifier accuracy 0.7482993197278912
AdaClassifier accuracy 0.8024691358024691
LogisticRegression  accuracy 0.8090199042579995
MLPClassifier accuracy 0.8115394305870496
KNeighborsClassifier accuracy 0.7377173091458806
KerasClassifier: test loss, test acc: [0.43896231055259705, 0.8022171854972839]


check the amount of wins and losses are in the training label data (should be almost equal):
1    8439
0    7434

```
### Correlation Matrix
![](https://github.com/bszek213/college_football_machine_learning/blob/master/correlations.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
