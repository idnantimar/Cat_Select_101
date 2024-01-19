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
from .. import My_Template_FeatureImportance




class LaplacianScore_importance(My_Template_FeatureImportance):
    """
        Featute selection using Laplacian Score based unsupervised method.

        Key Idea : When class-labels are known a feature is important if it has
        high `between-class` variability than low `within-class` variability. In
        unsupervised setup a nearest neighbor graph can be eployed to appriximate
        the locality structure.

        Note : This method can be considered as an unsupervised approximation of Fisher Score.

        Pros : Can be applied in unsupervised learning case.

        Cons : Importance can vary for different choice of `n_neighbors`. Also unlike
        Fisher Score, Laplacian Score is not scale invariant, since the `kneighbor_graph_`
        can vary with different scales.

        Parameters
        ----------
        max_features : int ; default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

        threshold : float ; default 1e-10
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise will be rejected.

        Attribures
        ----------

        References
        ----------
        ..[1] He, Xiaofei, Deng Cai, and Partha Niyogi. "Laplacian score for feature selection."
        Advances in neural information processing systems 18 (2005).

    """
    _coef_to_importance = None
    _permutation_importance = None
    def __init__(self,*,
                 max_features=None,threshold=1e-10):
        super().__init__()
        self.threshold = threshold
        self.max_features = max_features

    def _similarity(self,dist_matrix):
        """
        A function that returns similarity scores between observations,
        based on precomputed distance matrix. Default if RBF median heuristic.

        [ For internal use only. Override if necessary.]

        """
        sigma = np.median(dist_matrix[dist_matrix.astype(bool)])
        scale_dist_sq = (dist_matrix/sigma)**2
        return np.exp(-scale_dist_sq)


    def fit(self,X,*,n_neighbors=20,metric='euclidean',
            **kwargs_knn):
        """
        ``fit`` method for ``LaplacianScore_importance``.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        n_neighbors : int ; default 20
            Number of neighbors to use.

        metric : str ; default 'euclidean'
            The metric to use in ``sklearn.metrics.pairwise_distances()``.

        **kwargs_knn : other keyword arguments to ``sklearn.neighbors.NearestNeighbors()``.

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
        dist_matrix = pairwise_distances(X,metric=metric)
        ### knn graph .....
        ## node i & j are connected when atleast one is knn of the other
        KNN = NearestNeighbors(n_neighbors=n_neighbors,metric='precomputed',
                               **kwargs_knn)
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
        self.feature_importances_ = np.apply_along_axis(for_jColumn,
                                                        axis=0,arr=X)
        return self


    def transform(self,X):
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        return super().transform(X)


    def get_error_rates(self,true_coef,*,plot=False):
        """
        Computes various error-rates when true importance of the features are known.

        Parameters
        ----------
        true_coef : array of shape (`n_features_in_`,)
            If a boolean array , True implies the feature is important in true model, null feature otherwise.
            If a array of floats , it represent the `feature_importances_` of the true model.

        plot : bool ; default False
            Whether to plot the `confusion_matrix_for_features_`.

        Returns
        -------
        dict
            Returns the empirical estimate of various error-rates
           {'PCER': per-comparison error rate,
            'FDR': false discovery rate,
            'PFER': per-family error rate,
            'TPR': true positive rate
            }

        """
        self.get_support()
        self.true_coef = np.array(true_coef)
        if (self.true_coef.dtype==bool) :
            self.true_support = self.true_coef
        else :
            self.true_support = (self.true_coef >= self.threshold_)
        return super().get_error_rates(plot=plot)



#### ==========================================================================
