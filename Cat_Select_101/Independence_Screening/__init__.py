"""
Created on Thu Jan 11 23:46:38 2024

Topic: Usually variable selection methods rely on the coefficients in some parametric model of Y|X=x.
       But for Categorical Response case, Nonparametric tests for homogeneity can be applied for variable selection.
       This methods can specially work well for misspecified models.

@author: R.Nandi
"""



#### Model-Free approach to variable selection (for Continuous Predictors) ====

import numpy as np
import pandas as pd
from .. import My_Template_FeatureImportance,_Data_driven_Thresholding



class SIS_importance(My_Template_FeatureImportance):
    """
        Key Idea : In the setup of categorical response Y, a feature Xj is unimportant
        if Xj|Y=y are very similar for all categories y.

        Pros : Can be beneficial for misspecified models.

        Cons : This method only uses marginal utilities, ignoring joint covariance structure
        of the features.

        Parameters
        ----------
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

        Attribures
        ----------
        classes_ : array of shape (`n_classes_`,)
            A list of class labels known to the classifier.

        confusion_matrix_for_features_ : array of shape (`n_features_in_`, `n_features_in_`)
            ``confusion_matrix`` (`true_support`, `support_`)

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

        n_classes_ : int
            Number of target classes.

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
        ..[1] Cui, Hengjian, Runze Li, and Wei Zhong. "Model-free feature screening
        for ultrahigh dimensional discriminant analysis."
        Journal of the American Statistical Association 110.510 (2015): 630-641.

    """

    _coef_to_importance = None
    _permutation_importance = None
    def __init__(self,*,max_features=None,threshold=0,cumulative_score_cutoff=0.05):
        super().__init__()
        self.threshold = threshold
        self.max_features = max_features
        self.cumulative_score_cutoff = cumulative_score_cutoff


    def fit(self,X,y):
        """
        ``fit`` method for ``SIS_importance``.

        This method is extremely fast as it does not involve any cross-validation or hyperparameter tuning.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples.

            [ NOTE: only valid for numerical features ]

        y : Series of shape (n_samples,)
            The target values.

        Returns
        -------
        self
            The fitted ``SIS_importance`` instance is returned.
        """
        super().fit(X,y)
        ### empirical pmf of categorical response .....
        u = pd.get_dummies(y,dtype=float,drop_first=False).to_numpy().T
                ## each row is indicator for a response class label
        class_probs = u.mean(axis=-1,keepdims=True)
        ### computation for each feature .....
        u = (u/class_probs) - 1
                ## this will be useful for calculating [F(X|Y) - F(X)]
        def for_jColumn(Xj):
            ## E_x[Var_y(F(Xj|Y))]
            out = 0
            for x in Xj :
                v = np.mean((Xj<=x)*u,axis=-1,keepdims=True)
                    ## this quantity is [F(X|Y) - F(X)]
                out += np.sum(class_probs*(v**2),axis=None)
            return out/self.n_samples_
        ### iterating over the columns .....
        X = X.to_numpy()
        self.feature_importances_ = np.apply_along_axis(for_jColumn,
                                                        axis=0,arr=X)
        _Data_driven_Thresholding(self)
        return self


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
##
###
####
###
##
#### For Categorical Predictors ===============================================


class SIScat_importance(My_Template_FeatureImportance):
    """
        Key Idea : In the setup of categorical response Y, a feature Xj is unimportant
        if Xj|Y=y are very similar for all categories y.

        Pros : Can be beneficial for misspecified models.

        Cons : This method only uses marginal utilities, ignoring joint covariance structure
        of the features.

        Parameters
        ----------
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

        Attribures
        ----------
        classes_ : array of shape (`n_classes_`,)
            A list of class labels known to the classifier.

        confusion_matrix_for_features_ : array of shape (`n_features_in_`, `n_features_in_`)
            ``confusion_matrix`` (`true_support`, `support_`)

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

        n_classes_ : int
            Number of target classes.

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
        ..[1] Cui, Hengjian, Runze Li, and Wei Zhong. "Model-free feature screening
        for ultrahigh dimensional discriminant analysis."
        Journal of the American Statistical Association 110.510 (2015): 630-641.

        [ Just like CDF for continuous features, we will use PMF for categorical features ]

    """

    _coef_to_importance = None
    _permutation_importance = None
    def __init__(self,*,max_features=None,threshold=1e-10,cumulative_score_cutoff=0.05):
        super().__init__()
        self.threshold = threshold
        self.max_features = max_features
        self.cumulative_score_cutoff = cumulative_score_cutoff


    def fit(self,X,y):
        """
        ``fit`` method for ``SIScat_importance``.

        This method is extremely fast as it does not involve any cross-validation or hyperparameter tuning.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples.

            [ NOTE: only valid for categorical features ]

        y : Series of shape (n_samples,)
            The target values.

        Returns
        -------
        self
            The fitted ``SIScat_importance`` instance is returned.
        """
        super().fit(X,y)
        ### empirical pmf of categorical response .....
        u = pd.get_dummies(y,dtype=float,drop_first=False).to_numpy().T
                ## each row is indicator for a response class label
        class_probs = u.mean(axis=-1,keepdims=True)
        ### computation for each feature .....
        u = (u/class_probs) - 1
                ## this will be useful for calculating [P(X|Y) - P(X)]
        def for_jColumn(Xj):
            ## E_x[Var_y(P(Xj|Y))]
            Xj_ = pd.get_dummies(Xj,dtype=bool,drop_first=False)
                # each column is indicator for a predictor class label
            Xj_probs = Xj_.mean(axis=0)
                # marginal pmf for Xj
            out = np.apply_along_axis(_Var_Px_given_Y,axis=0,arr=Xj_)
            return out.dot(Xj_probs)
        def _Var_Px_given_Y(x):
            ## variance(P(x|Y)) for a fixed category of Xj
            v = np.mean(x*u,axis=-1,keepdims=True)
            return np.sum(class_probs*(v**2),axis=None)
        ### iterating over the columns .....
        self.feature_importances_ = X.apply(for_jColumn,axis=0).to_numpy()
        _Data_driven_Thresholding(self)
        return self


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
