# %%
import numpy as np 
import time
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold

from data import get_dataset
from pvoting import ProbabilisticVotingClassifier

# %% generate data
n = 1000
d = 20
d_inform = 15
k = 2
seed = 0
X, y = get_dataset(n, d, k, d_inform=d_inform, seed=seed)
print('data shape:', X.shape, y.shape)

# %% define base classifier models
base_models = {
    'knn-1': KNeighborsClassifier(n_neighbors=1),
    'knn-3': KNeighborsClassifier(n_neighbors=3),
    'knn-5': KNeighborsClassifier(n_neighbors=5),
    'knn-7': KNeighborsClassifier(n_neighbors=7),
    'knn-9': KNeighborsClassifier(n_neighbors=9),
    'svm-rbf': SVC(probability=True, kernel='rbf'),
    'svm-lin': SVC(probability=True, kernel='linear'),
    'svm-pol1': SVC(probability=True, kernel='poly', degree=1),
    'svm-pol2': SVC(probability=True, kernel='poly', degree=2),
    'svm-pol3': SVC(probability=True, kernel='poly', degree=3),
    'svm-pol4': SVC(probability=True, kernel='poly', degree=4),
    'svm-pol5': SVC(probability=True, kernel='poly', degree=5),
}
# define ensemble voting model
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
hard_voting_ensemble = VotingClassifier(
    estimators=[(name, clone(model)) for name,model in base_models.items()], 
    voting='hard',
)
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
base_estimator = clone(base_models['knn-5']) 
bagging_ensemble = BaggingClassifier(
    base_estimator,
    n_estimators=len(base_models),
)
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
randforest_ensemble = RandomForestClassifier(
    n_estimators=100,
    max_depth=2,
    random_state=0,
)
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
base_estimator = clone(base_models['svm-rbf']) 
adaboost_ensemble = AdaBoostClassifier(
    base_estimator, 
    n_estimators=len(base_models), 
    random_state=0
)
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
gradboost_ensemble = GradientBoostingClassifier(
    n_estimators=100, 
    learning_rate=1.0, 
    max_depth=2, 
    random_state=0
)

randomvoter_ensemble = ProbabilisticVotingClassifier(
    estimators=[(name, clone(model)) for name,model in base_models.items()], 
    voting='random',
)

mev0voter_ensemble = ProbabilisticVotingClassifier(
    estimators=[(name, clone(model)) for name,model in base_models.items()], 
    voting='mev0',
    mev0_kwargs={'Niters': 10, 'Nparticles': 20}
)



ensemble_models = {
    'hardvote': hard_voting_ensemble,
    'bagging': bagging_ensemble,
    'rforest': randforest_ensemble,
    'adaboost': adaboost_ensemble,
    'gradboost': gradboost_ensemble,
    'randomvote': randomvoter_ensemble, 
    'mev0vote': mev0voter_ensemble,
}


models = {**base_models, **ensemble_models}

# %% evaluate each model using cross-validation
def evaluate(model, X, y, n_splits=10, n_repeats=3, seed=seed):
    # scoring metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

results, names = list(), list()
rj = max(len(k) for k in models.keys())
for name,model in models.items():
    _st = time.perf_counter()
    scores = evaluate(model, X, y)
    _et = time.perf_counter()
    print(f'{name.rjust(10, " ")}: mean={np.mean(scores):.3f} std={np.std(scores):.3f} time={_et-_st:.4f}s')
    results.append(scores)
    names.append(name)

# %% plot model performance for comparison
def create_plot(results, names):
    fig, ax = plt.subplots()
    ax.boxplot(results, labels=names, showmeans=True)
    ax.tick_params(axis='x', labelrotation=90)
    return fig, ax

fig, ax = create_plot(results, names)
fig.savefig('./compare.png', bbox_inches='tight')