"""
Created on Sat Feb  3 16:52:18 2024

Topic: Relief based feature selection algorithms.

@author: R.Nandi
"""



#### Relief based importance ==================================================

import numpy as np
import pandas as pd
from skrebate import MultiSURF
from sklearn.preprocessing import OrdinalEncoder
from .. import My_Template_FeatureImportance,_Data_driven_Thresholding
from .._utils.generate_Categorical import detect_Categorical



class ReBATE_importance(My_Template_FeatureImportance):
    """
        Relief based feature selection.

        Key Idea : A good feature should have same value for instances from the
        same class and should differentiate between instances from different classes.

        Class Variables
        ---------------
        Estimator : ``skrebate.MultiSURF``

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

        estimator : a fitted ``skrebate.MultiSURF(...)`` instance

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

        intercept_ : array of shape (`n_classes_`,)
            Intercept added to the decision function.

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
        ..[1] Kira, Kenji, and Larry A. Rendell. "A practical approach to feature selection." Machine learning proceedings 1992.
        Morgan Kaufmann, 1992. 249-256.

        ..[2] Urbanowicz, Ryan J., et al. "Benchmarking relief-based feature selection methods for bioinformatics data mining."
        Journal of biomedical informatics 85 (2018): 168-188.


    """
    Estimator = MultiSURF
    _coef_to_importance = None
    _permutation_importance = None

    def __init__(self,*,
                 max_features=None,threshold=0,cumulative_score_cutoff=0.05,
                 set_params={'n_jobs':1,'verbose':False}):
        self.max_features = max_features
        self.threshold = threshold
        self.cumulative_score_cutoff = cumulative_score_cutoff
        self.set_params = set_params


    def _discrete_threshold(self,X):
        """
        Detects categorical variables.

        [ for internal use only ]

        """
        is_Cat = detect_Categorical(X)
        if any(is_Cat):
            count = X.iloc[:,is_Cat].apply(lambda s :len(np.unique(s)),axis=0)
            X = X.copy()
            X.iloc[:,is_Cat] = OrdinalEncoder().fit_transform(X.iloc[:,is_Cat])
            return np.max(count), X.to_numpy(dtype=float)
        else : return 1, X.to_numpy()


    def fit(self,X,y):
        """
        ``fit`` method for ``ReBATE_importance``.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : Series of shape (n_samples,)
            The target values.

        Returns
        -------
        self
            The fitted ``ReBATE_importance`` instance is returned.

        """
        X = pd.DataFrame(X)
        super().fit(X,y)
        discrete_threshold_,X_ = self._discrete_threshold(X)
        self.estimator = self.Estimator(n_features_to_select=(self.max_features
                                                         if isinstance(self.max_features,int) else self.n_features_in_),
                                   discrete_threshold=discrete_threshold_,
                                   **self.set_params)
        self.estimator.fit(X_,y)
        self.feature_importances_ = self.estimator.feature_importances_
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
