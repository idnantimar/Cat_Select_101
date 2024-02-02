"""
Created on Fri Dec 15 11:23:30 2023

Topic: LASSO based Variable Selection, using (i)multinomial loss and (ii)One-vs-Rest scheme.
       Since the wide popularity of LASSO , it is used as a baseline for comparison here.

@author: R.Nandi
"""



#### Vanilla LASSO ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
import os
from datetime import datetime
from . import My_Template_FeatureImportance,_Data_driven_Thresholding




class vanillaLASSO_importance(My_Template_FeatureImportance):
    """
        Feature selection using Logistic LASSO regression.


        Class Variables
        ---------------
        Estimator_Type : ``{'penalty':'l1','solver':'saga'}``

        Parameters
        ----------
        random_state : int ; default None
            Seed for reproducible results across multiple function calls.

        multiclass : {'multinomial','ovr'} ; default
            How to handle more than two target classes.

        Cs : list ; default ``list(np.logspace(-2,2,5))``
            The inverse of regularization strength.

        max_iter : int ; default 1000
            Maximum number of iterations of the optimization algorithm.

        reduce_norm : non-zero int, inf, -inf ; default 1
            Order of the norm used to compute `feature_importances_` in the case where the `coef_` of the
            underlying Logistic Regression is of dimension 2. By default 'l1'-norm is being used.

        max_features : int ; default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

        threshold : float ; 0
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise rejected.

        cumulative_score_cutoff : float in [0,1) ; default 0.05
            Computes data-driven 'threshold' for selecting those features that contributes to top
            100*(1-cut_off)% feature importances. Result is not valid when all features
            are unimportant.

        set_params_LogisticReg : dict ; default {}
            dict of other keyword arguments to ``LogistiRegression()`` or ``LogistiRegressionCV()``

        Attribures
        ----------
        C_ : the best value for inverse of regularization strength among `Cs` for each class, when ``len(Cs)``>1

        category_specific_importances_ : array of shape (`n_classes_`, `n_features_in_`)
            Feature importances specific to each target class, only available when `multi_class` = `ovr`.

        classes_ : array of shape (`n_classes_`,)
            A list of class labels known to the classifier.

        coef_ : array of shape (`n_classes_`, `n_features_in_`)
            Coefficient of the features in the decision function.

        confusion_matrix_for_features_ : array of shape (`n_features_in_`, `n_features_in_`)
            ``confusion_matrix`` (`true_support`, `support_`)

        estimator : a fitted ``LogisticRegression(...)`` or ``LogisticRegressionCV(...)`` instance

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

    """
    Estimator_Type = {'penalty':'l1','solver':'saga'}

    def __init__(self,random_state=None,*,multi_class='multinomial',Cs=list(np.logspace(-4,4,10)),
                 max_iter=1000,reduce_norm=1,
                 max_features=None,threshold=0,cumulative_score_cutoff=0.05,
                 set_params_LogisticReg={}):
        super().__init__(random_state)
        self.multi_class = multi_class
        self.Cs = Cs
        self.max_iter = max_iter
        self.reduce_norm = reduce_norm
        self.max_features = max_features
        self.threshold = threshold
        self.cumulative_score_cutoff = cumulative_score_cutoff
        self.set_params_LogisticReg = set_params_LogisticReg


    def fit(self,X,y):
        """
        ``fit`` method for ``vanillaLASSO_importance``.

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
            The fitted ``vanillaLASSO_importance`` instance is returned.

        """
        X = pd.get_dummies(pd.DataFrame(X),drop_first=True,dtype=int)
        super().fit(X,y)
        ### fitting the model .....
        self.Estimator_Type.update(self.set_params_LogisticReg)
        if len(self.Cs)>1 :
            estimator = LogisticRegressionCV(**self.Estimator_Type,
                                             multi_class=self.multi_class,
                                             Cs=self.Cs,
                                             random_state=self.random_state,max_iter=self.max_iter,
                                             )
            estimator.fit(X,y)
            self.C_ = estimator.C_
        else :
            estimator = LogisticRegression(**self.Estimator_Type,
                                           multi_class=self.multi_class,
                                           C=self.Cs[0],
                                           random_state=self.random_state,max_iter=self.max_iter,
                                           )
            estimator.fit(X,y)
        ### feature_importances .....
        self.estimator = estimator
        self.coef_ = estimator.coef_
        self.intercept_ = estimator.intercept_
        if self.multi_class=='multinomial' :
            self.feature_importances_ = self._coef_to_importance(self.coef_,self.reduce_norm,
                                                             identifiability=False)
        else :
            self.feature_importances_ = self._coef_to_importance(self.coef_,self.reduce_norm,
                                                             identifiability=True)
            self.category_specific_importances_ = np.abs(self.coef_)**self.reduce_norm
        ### ranking and threshold .....
        _Data_driven_Thresholding(self)
        return self


    def get_permutation_importances(self,test_data,*,n_repeats=10):
        """
        Key Idea : Fit a model based on all features, then every time randomly permute observations of one feature column,
        keeping the other columns fixed, to break the association between that feature and response. Evaluate the
        performance of the fitted model once on permuted data and once on unpermuted data. The more
        important a feature is, the larger will be the corresponding drop in performance after permutation.

        Note : It implicitly assumes independence of features. Not suitable
        when features are highly correlated or one-hot-encoded categorical features are there.

        Parameters
        ----------
        test_data : tuple (X_test,y_test)
            X_test has shape (n_samples,n_features)
            and y_test has shape (n_samples,)

        n_repeats : int ; default 10
            Number of times to permute a feature.

        Returns
        -------
        A ``sklearn.inspection.permutation_importance`` object.


        [ Calling this function will override the `coef_` based `feature_importances_` ]

        """
        X_test,y_test = test_data
        X_test = pd.get_dummies(pd.DataFrame(X_test),drop_first=True,dtype=int)
        out = super()._permutation_importance((X_test,y_test),n_repeats=n_repeats,
                                               scoring=None)
        _Data_driven_Thresholding(self)
        return out


    def transform(self,X):
        X = pd.get_dummies(pd.DataFrame(X),drop_first=True,dtype=int)
        return super().transform(X)


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


    def plot(self,sort=True,savefig=None,**kwargs):
        kwargs.update({'ylabel':'vanillaLASSO_'+str(self.multi_class)})
        super().plot(sort=sort,savefig=savefig,**kwargs)


    def dump_to_file(self,file_path=None):
        if file_path is None :
            file_path = os.path.join(os.getcwd(),
                                     f"vanillaLASSO_{self.multi_class}-{datetime.now().strftime('%Y_%m_%d_%H%M%S%f')}.pkl")
        super().dump_to_file(file_path)





#### ==========================================================================
