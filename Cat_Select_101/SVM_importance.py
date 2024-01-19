"""
Created on Wed Jan 17 15:18:59 2024

Topic: We can use SVM with One-vs-Rest scheme for feature selection.

@author: R.Nandi
"""



#### Multiclass L1SVM =========================================================

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from . import My_Template_FeatureImportance



class L1SVM_importance(My_Template_FeatureImportance):
    """
        Feature selection based on 1-norm Support Vector Machines (SVM).

        [ Usually SVM is defined with 'l2'-penalties. But using 'l1'-penalties leads
        to sparser `coef_`. ]

        Pros : This method gives response-category specific feature importances.

        Cons : `n_classes` separate models are needed to be trained.

        Class Variables
        ---------------
        Estimator_Type : ``LinearSVC(penalty='l1',dual=False,multi_class='ovr')``

        Parameters
        ----------
        random_state : int ; default None
            Seed for reproducible results across multiple function calls.

        max_features : int ; default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

        threshold : float ; default 1e-10
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise will be rejected.

        References
        ----------
        ..[1] Zhu, Ji, et al. "1-norm support vector machines." Advances in
        neural information processing systems 16 (2003).

    """
    Estimator_Type = LinearSVC(penalty='l1',dual=False,multi_class='ovr')

    def __init__(self,random_state=None,*,
                 max_features=None,threshold=1e-10):
        super().__init__(random_state)
        self.max_features=max_features
        self.threshold=threshold


    def fit(self,X,y,Cs=[1.0],*,
            cv_config={'cv':None,'n_jobs':None,'verbose':2},
            reduce_norm=1):
        """
        ``fit`` method for ``L1SVM_importance``

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : Series of shape (n_samples,)
            The target values.

        Cs : list ; default [1.0]
            The inverse of the regularization strength.

        cv_config : dict of keyword arguments to ``GridSearchCV`` ; default ``{'cv':None,'n_jobs':None,'verbose':2}``
            Will be used when `Cs` has to be determined by crossvalidation.

        reduce_norm : non-zero int, inf, -inf ; default 1
            Order of the norm used to compute `feature_importances_` in the case where the `coef_` of the
            underlying SVM is of dimension 2. By default 'l1'-norm is being used.

        Returns
        -------
        self
            The fitted ``L1SVM`` instance is returned.

        """
        X = pd.get_dummies(X,dtype=int,drop_first=True)
        super().fit(X,y)
        ### assigning the estimator .....
        estimator = clone(self.Estimator_Type)
        estimator.set_params(C=Cs[0],random_state=self.random_state)
        ### fitting the model .....
        if len(Cs)>1 :
            cv_config.update({'refit':True})
            self.gridsearch = GridSearchCV(estimator,
                                           param_grid={'C':Cs},
                                           **cv_config)
            self.gridsearch.fit(X,y)
            self.estimator = self.gridsearch.best_estimator_
            self.C_ = self.gridsearch.best_params_['C']
        else :
            estimator.fit(X,y)
            self.estimator = estimator
            self.C_ = Cs[0]
        ### feature_importances .....
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        self.feature_importances_ = self._coef_to_importance(self.coef_,
                                                             reduce_norm,identifiability=True)
        self.category_specific_importances_ = np.abs(self.coef_)**reduce_norm
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
            y_test has shape (n_samples,)

        n_repeats : int ; default 10
            Number of times to permute a feature.

        Returns
        -------
        A ``sklearn.inspection.permutation_importance`` object.


        [ Calling this function will override the `coef_` based `feature_importances_` ]

        """
        X_test,y_test = test_data
        X_test = pd.get_dummies(X_test,drop_first=True,dtype=int)
        return super()._permutation_importance((X_test,y_test),
                                               n_repeats=n_repeats,
                                               scoring=None)


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





