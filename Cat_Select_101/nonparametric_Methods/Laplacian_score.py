"""
Created on Fri Jan 19 16:07:02 2024

Topic:  In many real world classification problems, data from the same class are often close to each other.
        The importance of a feature can be evaluated by its power of locality preserving.

@author: R.Nandi
"""



#### Laplacian Score based importance =========================================


import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from .. import My_Template_FeatureImportance,_Data_driven_Thresholding




class LaplacianScore_importance(My_Template_FeatureImportance):
    """
        Featute selection using Laplacian Score based unsupervised method.

        Key Idea : When class-labels are known a feature is important if it has
        high `between-class` variability than low `within-class` variability. In
        unsupervised setup a nearest neighbor graph can be eployed to appriximate
        the locality structure.

        Note : This method can be considered as an unsupervised approximation of Fisher Score.

        Pros : Can be applied in unsupervised learning case.

        Cons : Importances can vary for different choice of `n_neighbors`. Also unlike
        Fisher Score, Laplacian Score is not scale invariant, since the `kneighbors_graph_`
        can vary with varying scales of features.

        Parameters
        ----------
        n_neighbors : int ; default 20
            Number of neighbors to use.

        metric : str ; default 'euclidean'
            The metric to use in ``sklearn.metrics.pairwise_distances()``.

        max_features : int ; default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

        threshold : float ; default 0
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise will be rejected.

        cumulative_score_cutoff : float in [0,1) ; default 0.05
            Computes data-driven 'threshold' for selecting those features that contributes to top
            100*(1-cut_off)% feature importances. Result is not valid when all features
            are unimportant.

        kwargs_knn : dict ; default {}
            dict of other keyword arguments to ``sklearn.neighbors.NearestNeighbors()``.


        Attribures
        ----------
        confusion_matrix_for_features_ : array of shape (`n_features_in_`, `n_features_in_`)
            ``confusion_matrix`` (`true_support`, `support_`)

        estimator : a fitted ``NearsestNeighbors(...)`` instance

        false_discoveries_ : array of shape (`n_features_in_`,)
            Boolean mask of false positives.

        false_negatives_ : array of shape (`n_features_in_`,)
            Boolean mask of false negatives.

        feature_importances_ : array of shape (`n_features_in_`,)
            Importances of features.

        feature_names_in_ : array of shape (`n_features_in_`,)
            Names of features seen during ``fit``.

        features_selected_ : array of shape (`n_features_selected_`,)
            Names of selected features.

        kneighbours_graph_ : csr matrix of shape (`n_samples_`, `n_samples_`)
            [i,j] element is True if i-th sample is neighbor of j-th sample or
            j-th sample is neighbor of i-th sample, False otherwise.

        n_features_in_ : int
            Number of features seen during ``fit``.

        n_features_selected_ : int
            Number of selected features.

        n_samples_ : int
            Number of observations seen during ``fit``.

        ranking_ : array of shape (`n_features_in_`,)
            The feature ranking, such that ``ranking_[i]`` corresponds to the
            i-th best feature, i=1,2,..., `n_features_in_`.

        support_ : array of shape (`n_features_in_`,)
            Boolean mask of selected features.

        threshold_ : float
            Cut-off in use, for selection/rejection.

        true_support : array of shape (`n_features_in_`,)
            Boolean mask of active features in population, only available after
            ``get_error_rates`` method is called with true_imp.

        References
        ----------
        ..[1] He, Xiaofei, Deng Cai, and Partha Niyogi. "Laplacian score for feature selection."
        Advances in neural information processing systems 18 (2005).

    """
    _coef_to_importance = None
    _permutation_importance = None
    def __init__(self,*,n_neighbors=20,metric='euclidean',
                 max_features=None,threshold=0,cumulative_score_cutoff=0.05,
                 kwargs_knn={}):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.max_features = max_features
        self.threshold = threshold
        self.cumulative_score_cutoff = cumulative_score_cutoff
        self.kwargs_knn = kwargs_knn


    def _similarity(self,dist_matrix):
        """
        A function that returns similarity scores between observations,
        based on precomputed distance matrix. Default if RBF median heuristic.

        [ For internal use only. Override if necessary.]

        """
        sigma = np.median(dist_matrix[dist_matrix.astype(bool)])
        scale_dist_sq = (dist_matrix/sigma)**2
        return np.exp(-scale_dist_sq)


    def fit(self,X,y=None):
        """
        ``fit`` method for ``LaplacianScore_importance``.

        [(1-L)/L -1] is used as feature_importances here.


        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : None
            Ignored.

        Returns
        -------
        self
            The fitted ``LaplacianScore_importance`` instance is returned.

        """
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        self.feature_names_in_ = X.columns
        self.n_samples_,self.n_features_in_ = X.shape
        X = X.to_numpy()
        ### precomputed distance matrix .....
        dist_matrix = pairwise_distances(X,metric=self.metric)
        ### knn graph .....
        ## node i & j are connected when atleast one is knn of the other
        KNN = NearestNeighbors(n_neighbors=self.n_neighbors,metric='precomputed',
                               **self.kwargs_knn)
        KNN.fit(dist_matrix)
        affinity = KNN.kneighbors_graph(mode='connectivity')
        self.kneighbours_graph_ = (affinity + affinity.T).astype(bool)
        ### similarity scores .....
        S = self._similarity(dist_matrix)
        S *= self.kneighbours_graph_.toarray()
        ### Laplacian Score .....
        D = np.sum(S,axis=1,keepdims=False)
        def for_jColumn(Xj):
            Xj -= np.dot(Xj,D)/np.sum(D)
            if any(Xj) :
                ## smaller Lapcian Score Lr=f(D-S)f/fDf is preferable for a feature, possible value (0,1)
                ## so (1/Lr - 1) will be used as feature_importances
                ## that is our importance is f(D-(D-S))f/f(D-S)f = fSf/f(D-S)f
                fDf = np.dot(D,Xj**2)
                fSf = np.dot(Xj,S).dot(Xj)
                return fSf/(fDf-fSf)
            else :
                ## Xj is a constant feature, so trivially rejectet
                return 0

        ### iterating over the columns .....
        self.estimator = KNN
        self.feature_importances_ = -1 + np.apply_along_axis(for_jColumn,
                                                        axis=0,arr=X)
        _Data_driven_Thresholding(self)
        return self


    def transform(self,X):
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        return super().transform(X)


    def fit_transform(self,X,y=None):
        """
        ``fit`` the data (X,y) then ``transform`` X.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples.

        y : None
            Ignored.

        Returns
        -------
        DataFrame of shape (n_samples, n_selected_features)
            The input samples with only the selected features.

        """
        super().fit_transform(X,None)


    def get_error_rates(self,true_imp,*,plot=False):
        """
        Computes various error-rates when true importance of the features are known.

        *   If a feature is True in `support_` and False in `true_support`
            it is a false-discovery or false +ve

        *   If a feature is False in `support_` and True in `true_support`
            it is a false -ve

        Parameters
        ----------
        true_imp : array of shape (`n_features_in_`,)
            If a boolean array , True implies the feature is important in true model, null feature otherwise.
            If an array of floats , it represent the `feature_importances_` of the true model.

        plot : bool ; default False
            Whether to plot the `confusion_matrix_for_features_`.

        Returns
        -------
        dict
            Returns the empirical estimate of various error-rates
           {
               'PCER': per-comparison error rate ; ``mean`` (`false_discoveries_`),

               'FDR': false discovery rate ; 1 - ``precision`` (`true_support`, `support_`),

               'PFER': per-family error rate ; ``sum`` (`false_discoveries_`),

               'TPR': true positive rate ; ``recall`` (`true_support`, `support_`),

               'n_FalseNegatives': number of false -ve ;  ``sum`` (`false_negatives_`),

               'minModel_size': maximum rank of important features ; ``max`` (`ranking_` [ `true_support` ]),

               'selection_F1': ``F1_score`` (`true_support`, `support_`),

               'selection_YoudenJ': ``sensitivity`` (`true_support`, `support_`) + ``specificity`` (`true_support`, `support_`) - 1
            }

        """
        self.get_support()
        true_imp = np.array(true_imp)
        if (true_imp.dtype==bool) :
            self.true_support = true_imp
        else :
            self.true_support = (true_imp > self.threshold_)
        return super()._get_error_rates(plot=plot)



#### ==========================================================================

