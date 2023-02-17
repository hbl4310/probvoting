from sklearn.datasets import make_classification

def get_dataset(n, d, k, d_inform=None, seed=0):
    d_inform = d_inform if not d_inform is None else d
    X, y = make_classification(
        n_samples=n, 
        n_features=d, 
        n_informative=d_inform, 
        n_redundant=d-d_inform, 
        n_classes=k,
        random_state=seed,
    )
    return X, y