"""
Created on Thu Jan 18 00:52:15 2024

Topic: Feature Importance Based on Discriminative Least Squares Regression.

@author: R.Nandi
"""



#### Discriminative Least Squares Regression for feature selection ============

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from .. import My_Template_FeatureImportance
from .._utils._d_lsr import d_LSR



class dLS_impotance(My_Template_FeatureImportance):
    """
        Feature selection based on coefficients of least squares regression.

        [ Least Squares method is commonly used in regression setup with continuous response.
        But introducing slack variables it can be extended to classification setup. ]

        Pros : Instead of One-vs-One or One-vs-Rest schemes, here we have one single model to train.

        Cons : Unlike ``LogisticRegression``, this method does not inherently provide probability
        estimates of the predicted classes.

        Parameters
        ----------
        max_features : int ; default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

        threshold : float ; default 1e-10
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise will be rejected.

        References
        ----------
        ..[1] Xiang, Shiming, et al. "Discriminative least squares regression for multiclass
        classification and feature selection." IEEE transactions on neural networks and learning
        systems 23.11 (2012): 1738-1754.


    """
    def __init__(self,*,max_features=None,threshold=1e-10):
        super().__init__()
        self.max_features = max_features
        self.threshold = threshold


    def fit(self,X,y,regularization=[0.1],*,initial_guess=(None,None),
            cv_config={'cv':None,'n_jobs':None,'verbose':2},max_iter=30,
            reduce_norm=2,u=1e+4,tol=1e-4):
        """
        ``fit`` method for ``dLS_impotance``.

        Calling this method will start iterations from beginning everytime. To resume an
        existing run, proceed as follows -

            >>> Model.fit(X,y,...) # fitting for the first time with required parameters
            >>> X_,y_ = pd.get_dummies(X,drop_first=True),pd.get_dummies(y)
            >>> Model.estimator.fit(X_,y_,warm_start=True) # resuming previous run
            >>> Model.update_importance() # update feature_importances_


        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : Series of shape (n_samples,)
            The target values.

        regularization : list ; default [0.1]
            The strength of regularization.

        initial_guess : tuple (W0,b0)
            Where,
            W0 is array of shape (n_features,n_classes)
            and b0 is array of shape (1,n_classes)

            When None, will be initialized at ``np.zeros()``.

        cv_config : dict of keyword arguments to ``GridSearchCV`` ; default ``{'cv':None,'n_jobs':None,'verbose':2}``
            Will be used when `regularization` has to be determined by crossvalidation.

        max_iter : int ; default 30
            The maximum number of iterations.

        reduce_norm : non-zero int, inf, -inf ; default 2
            Order of the norm used to compute `feature_importances_` in the case where the `coef_` of the
            underlying model is of dimension 2. By default 'l2'-norm is being used.

        u : float ; default 1e+4
            A large positive number (virtually +ve infinity).

        tol : float ; default 1e-4
            The tolerence for convergence criterion.

        Returns
        -------
        self
            The fitted ``dLS_impotance`` instance is returned.

        """

        X = pd.get_dummies(X,dtype=float,drop_first=True)
        super().fit(X,y)
        y = pd.get_dummies(y,dtype=float,drop_first=False)
        ### assigning the estimator .....
        estimator = d_LSR(regularization[0],u)
        W0,t0 = initial_guess
        if W0 is None : W0 = np.zeros((self.n_features_in_,self.n_classes_),dtype=float)
        if t0 is None : t0 = np.zeros((1,self.n_classes_),dtype=float)
        ### fitting the Model .....
        fit_params = {'W0':W0,'t0':t0,'max_iter':max_iter,'tol':tol}
        if len(regularization)>1 :
            cv_config.update({'refit':True})
            self.gridsearch = GridSearchCV(estimator,
                                           param_grid={'regularization':regularization},
                                           **cv_config)
            self.gridsearch.fit(X.to_numpy(),y.to_numpy(),**fit_params)
            self.estimator = self.gridsearch.best_estimator_
            self.best_penalty_ = self.gridsearch.best_params_['regularization']
        else :
            estimator.fit(X.to_numpy(),y.to_numpy(),**fit_params)
            self.estimator = estimator
            self.best_penalty_ = regularization[0]
        ### feature_importances .....
        self.coef_ = self.estimator.coef_.T
        self.intercept_ = self.estimator.intercept_.ravel()
        self._reduce_norm = reduce_norm
        self.feature_importances_ = self._coef_to_importance(self.coef_,
                                                             reduce_norm,identifiability=True)
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
        y_test =  pd.Categorical(y_test,categories=self.classes_)
        return super()._permutation_importance((X_test.to_numpy(),pd.get_dummies(y_test,dtype=int,drop_first=False).to_numpy()),
                                               n_repeats=n_repeats,
                                               scoring=None)


    def update_importance(self,imp_kind='coef',**kwargs_pimp):
        """
        After resuming an existing training run, update `feature_importances_`
        based on updated `coef_` and `intercept_`.

        Parameters
        ----------
        kind : a string from {'coef','permutation'} ; default 'coef'
            Which kind of feature importances to be updated.

        **kwargs_pimp : other keyword arguments to ``get_permutation_importances`` method.
        """
        self.coef_ = self.estimator.coef_.T
        self.intercept_ = self.estimator.intercept_.ravel()
        if imp_kind == 'permutation':
            self.get_permutation_importances(**kwargs_pimp)
        else :
            self.feature_importances_ = self._coef_to_importance(self.coef_,
                                                                 self._reduce_norm,identifiability=True)


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

