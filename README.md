# College Football Game Predictions

Scripts that predict the outcome of any Division I college football game.

## Installation
```bash
conda env create -f cfb_env
```

## Usage

```python
python cfb_ml.py
```
### Current prediction accuracies
```bash
Removed features (>0.85 correlation):  ['rush_yds_per_att', 'first_down_pass', 'first_down_rush', 'penalty_yds']

GradientBoostingClassifier - best params:  {'criterion': 'friedman_mse', 'learning_rate': 0.2, 'loss': 'deviance', 'max_depth': 2, 'max_features': 'auto', 'n_estimators': 400}
RandomForestClassifier - best params:  GridSearchCV(cv=5, estimator=RandomForestClassifier(),
             param_grid={'criterion': ['gini', 'entropy'],
                         'max_depth': array([1, 2, 3, 4]),
                         'max_features': ['auto', 'sqrt', 'log2'],
                         'n_estimators': range(100, 500, 100)},
             refit='accuracy', scoring=['accuracy'], verbose=4)
DecisionTreeClassifier - best params:  GridSearchCV(cv=5, estimator=DecisionTreeClassifier(),
             param_grid={'criterion': ['gini', 'entropy'],
                         'max_depth': array([1, 2, 3, 4]),
                         'max_features': ['auto', 'sqrt', 'log2'],
                         'splitter': ['best', 'random']},
             refit='accuracy', scoring=['accuracy'], verbose=4)
SVC - best params:  GridSearchCV(cv=5, estimator=SVC(),
             param_grid={'C': array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5]),
                         'gamma': ['scale', 'auto'],
                         'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                         'tol': array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])},
             refit='accuracy', scoring=['accuracy'], verbose=4)
LogisticRegression - best params: {'C': 4.5, 'max_iter': 500, 'penalty': 'l2', 'solver': 'sag'}
MLPClassifier - best params:  GridSearchCV(cv=5, estimator=MLPClassifier(),
             param_grid={'learning_rate': ['constant', 'invscaling',
                                           'adaptive'],
                         'learning_rate_init': array([0.001, 0.002, 0.003, 0.004]),
                         'max_iter': range(100, 1000, 100),
                         'solver': ['lbfgs', 'sgd', 'adam'],
                         'tol': array([0.001, 0.002, 0.003, 0.004])},
             refit='accuracy', scoring=['accuracy'], verbose=4)
KNeighborsClassifier - best params:  GridSearchCV(cv=5, estimator=KNeighborsClassifier(),
             param_grid={'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                         'n_neighbors': range(100, 1000, 100), 'p': [1, 2],
                         'weights': ['uniform', 'distance']},
             refit='accuracy', scoring=['accuracy'], verbose=4)

Gradclass 0.7983643542019176
RandForclass 0.7710095882684715
DecTreeclass 0.739424703891709
SVCclass 0.8056965595036661
LogReg 0.8040045121263395
MLPClass 0.81443880428652
KClass 0.7848279751833052
```
### Correlation Matrix
![](https://github.com/bszek213/college_football_machine_learning/blob/master/correlations.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.