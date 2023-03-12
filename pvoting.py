import numpy as np
import numbers
from mev0 import sim2

from sklearn.ensemble._voting import _BaseVoting
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch
from sklearn.utils import check_scalar
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _check_feature_names_in
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import column_or_1d
from sklearn.exceptions import NotFittedError
from sklearn.utils._estimator_html_repr import _VisualBlock
from sklearn.utils.fixes import delayed

# https://github.com/scikit-learn/scikit-learn/blob/1.1.X/sklearn/ensemble/_voting.py
class ProbabilisticVotingClassifier(ClassifierMixin, _BaseVoting):
    """Soft Voting/Majority Rule classifier for unfitted estimators.
    Read more in the :ref:`User Guide <voting_classifier>`.
    .. versionadded:: 0.17
    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'`` using
        :meth:`set_params`.
        .. versionchanged:: 0.21
            ``'drop'`` is accepted. Using None was deprecated in 0.22 and
            support was removed in 0.24.
    voting : {'hard', 'soft'}, default='hard'
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.
    weights : array-like of shape (n_classifiers,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.
    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        .. versionadded:: 0.18
    flatten_transform : bool, default=True
        Affects shape of transform output only when voting='soft'
        If voting='soft' and flatten_transform=True, transform method returns
        matrix with shape (n_samples, n_classifiers * n_classes). If
        flatten_transform=False, it returns
        (n_classifiers, n_samples, n_classes).
    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.
        .. versionadded:: 0.23
    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not 'drop'.
    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.
        .. versionadded:: 0.20
    le_ : :class:`~sklearn.preprocessing.LabelEncoder`
        Transformer used to encode the labels during fit and decode during
        prediction.
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying classifier exposes such an attribute when fit.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.
        .. versionadded:: 1.0
    See Also
    --------
    VotingRegressor : Prediction voting regressor.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    >>> clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    >>> clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> np.array_equal(eclf1.named_estimators_.lr.predict(X),
    ...                eclf1.named_estimators_['lr'].predict(X))
    True
    >>> eclf2 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...         voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    To drop an estimator, :meth:`set_params` can be used to remove it. Here we
    dropped one of the estimators, resulting in 2 fitted estimators:
    >>> eclf2 = eclf2.set_params(lr='drop')
    >>> eclf2 = eclf2.fit(X, y)
    >>> len(eclf2.estimators_)
    2
    Setting `flatten_transform=True` with `voting='soft'` flattens output shape of
    `transform`:
    >>> eclf3 = VotingClassifier(estimators=[
    ...        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...        voting='soft', weights=[2,1,1],
    ...        flatten_transform=True)
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>> print(eclf3.transform(X).shape)
    (6, 6)
    """

    def __init__(
        self,
        estimators,
        *,
        voting="hard",
        weights=None,
        n_jobs=None,
        flatten_transform=True,
        verbose=False,
        mev0_kwargs=dict(),
    ):
        super().__init__(estimators=estimators)
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose
        self.mev0_kwargs = mev0_kwargs

    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.
            .. versionadded:: 0.18
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        check_classification_targets(y)
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError(
                "Multilabel and multi-output classification is not supported."
            )

        check_scalar(
            self.flatten_transform,
            name="flatten_transform",
            target_type=(numbers.Integral, np.bool_),
        )

        # if self.voting not in ("soft", "hard"):
        #     raise ValueError(
        #         f"Voting must be 'soft' or 'hard'; got (voting={self.voting!r})"
        #     )

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        transformed_y = self.le_.transform(y)

        return super().fit(X, transformed_y, sample_weight)

    def predict(self, X):
        """Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        maj : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        if self.voting == "soft":
            maj = np.argmax(self.predict_proba(X), axis=1)

        elif self.voting == 'hard':  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self._weights_not_none)),
                axis=1,
                arr=predictions,
            )

        elif self.voting == 'random':
            n = X.shape[0]
            k = self.classes_.shape[0]
            c = len(self.estimators)
            probas = self._collect_probas(X)        # num_estimators x n x num_classes
            i = np.random.randint(0, c, (n,))       # for each n datapoints, choose a random estimator
            # TODO i'm sure we can vectorise this operation
            random_probas = np.zeros((n,k))
            for _n in range(n): 
                random_probas[_n, :] = probas[i[_n], _n, :]
            random_orderpos = np.argsort(-random_probas, axis=1)  # class rankings: classes [c0, c1, c2] ranked [1, 0, 2] with prob. scores p1 > p0 > p2
            random_orders = np.arange(k)[random_orderpos]         # class ordering: [c1, c0, c2]
            maj = random_orders[:, 0]

            # predictions = self._predict(X)
            # n,c = predictions.shape
            # i = np.random.randint(0, c, (n,))
            # maj = (predictions * np.eye(c, dtype=int)[i]).sum(axis=1)
        
        elif self.voting == 'randomseq':
            # for single top preference, this is the same as 'random'
            n = X.shape[0]
            k = self.classes_.shape[0]
            c = len(self.estimators)
            probas = self._collect_probas(X)        # num_estimators x n x num_classes
            i = np.random.randint(0, c, (n, k-1))    # sequence of random estimators drawn (with replacement) for all classes for each data point
            # TODO i'm sure we can vectorise this operation
            random_seq_probas = np.zeros((n,k-1,k))
            for _n in range(n): 
                random_seq_probas[_n, :, :] = probas[i[_n], _n, :]
            random_seq_orderpos_all = np.argsort(-random_seq_probas, axis=2)     # class rankings for each sequential estimator; n x k-1 x k
            random_seq_orders_all = np.arange(k)[random_seq_orderpos_all].reshape(n, (k-1)*k)
            random_seq_orders = np.apply_along_axis(
                            lambda x: resolve_sequential_orders(x, k),
                            axis=1,
                            arr=random_seq_orders_all.reshape(n, k*(k-1)),
                        )
            maj = random_seq_orders[:, 0] 

        elif self.voting == 'mev0':
            c = len(self.estimators)
            k = self.classes_.shape[0]
            probas = self._collect_probas(X)        # num_estimators x n x num_classes
            # pass in k x c probas (scaled by x10?) into marginstableorcreateseed
            probas_reshape = probas.transpose((1, 0, 2)).reshape(-1, c*k)   # n x (c*k)
            # _st = time.perf_counter()
            # np.random.seed(0) # for consistent results
            mev0_orders = np.apply_along_axis(
                lambda x: sim2(
                    marginstableorcreateseed=10.*x.reshape(c, k),   # scale scores for stability?
                    # suppress printing
                    debug_print=False, 
                    report_debug_interval=0,
                    # control accuracy vs time trade-off (TODO pass in parameters)
                    # Niters=20, 
                    # Nparticles=20,
                    seed=np.random.randint(2**31),
                    **self.mev0_kwargs,
                )[0],
                axis=1,
                arr=probas_reshape,
            )
            # _et = time.perf_counter()
            # print(f'{_et-_st:.4f}s')
            maj = mev0_orders[:, 0]


        maj = self.le_.inverse_transform(maj)

        return maj

    def _collect_probas(self, X):
        """Collect results from clf.predict calls."""
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

    def _check_voting(self):
        if self.voting == "hard":
            raise AttributeError(
                f"predict_proba is not available when voting={repr(self.voting)}"
            )
        return True

    @available_if(_check_voting)
    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        avg : array-like of shape (n_samples, n_classes)
            Weighted average probability for each class per sample.
        """
        check_is_fitted(self)
        avg = np.average(
            self._collect_probas(X), axis=0, weights=self._weights_not_none
        )
        return avg

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        probabilities_or_labels
            If `voting='soft'` and `flatten_transform=True`:
                returns ndarray of shape (n_samples, n_classifiers * n_classes),
                being class probabilities calculated by each classifier.
            If `voting='soft' and `flatten_transform=False`:
                ndarray of shape (n_classifiers, n_samples, n_classes)
            If `voting='hard'`:
                ndarray of shape (n_samples, n_classifiers), being
                class labels predicted by each classifier.
        """
        check_is_fitted(self)

        if self.voting == "soft":
            probas = self._collect_probas(X)
            if not self.flatten_transform:
                return probas
            return np.hstack(probas)

        else:
            return self._predict(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present here for API consistency by convention.
        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        if self.voting == "soft" and not self.flatten_transform:
            raise ValueError(
                "get_feature_names_out is not supported when `voting='soft'` and "
                "`flatten_transform=False`"
            )

        _check_feature_names_in(self, input_features, generate_names=False)
        class_name = self.__class__.__name__.lower()

        active_names = [name for name, est in self.estimators if est != "drop"]

        if self.voting == "hard":
            return np.asarray(
                [f"{class_name}_{name}" for name in active_names], dtype=object
            )

        # voting == "soft"
        n_classes = len(self.classes_)
        names_out = [
            f"{class_name}_{name}{i}" for name in active_names for i in range(n_classes)
        ]
        return np.asarray(names_out, dtype=object)


def resolve_sequential_orders(x, k):
    # take first preference from first candidate, next highest from second candidate... 
    # concatenate lower triangle of the square matrix: k (candidate orderings) x k (classes), 
    # where the last candidate is repeated since only k-1 degrees of freedom
    stacked_orders = np.concatenate([x[_i*k:_i*(k+1)+1] for _i in range(k-1)] + [x[-k:]])
    # return the index-preserved unique class labels
    _, idx = np.unique(stacked_orders, return_index=True)
    return stacked_orders[np.sort(idx)]

