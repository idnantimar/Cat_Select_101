"""
Created on Mon Jan  1 02:05:45 2024

Topic: Minimax Concave Penalty implementation for Logistic Regression beyond binary response.
       Upto binary response case, it is available on R package ``ncvreg`` [https://cran.r-project.org/web/packages/ncvreg/index.html] .

@author: R.Nandi
"""



#### MM algorithm based implementation ========================================

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from .. import My_Template_FeatureImportance



##> ..............................................................
from .._utils._WLasso_LogisticReg import WLasso_Logistic

class _LogisticMCPRegression_PyMc(BaseEstimator):
    """
        Logistic MCP regression using ``PyMC``.
    """
    def __init__(self,n_classes,strength=1,concavity=3,random_state=None):
        self.n_classes = n_classes
        ## number of target classes
        self.strength = strength
        self.concavity = concavity
        ## controls the strength and concavity of penalty
        self.random_state = random_state

    def _rhd_MCP(self,W):
        ## the right hand derivatives of MCP petalty wrt |W|
        ## we have rhd_MCP(0+)=l
        u = self.strength - np.abs(W)/self.concavity
        return np.where(u>0,u,0)

    def fit(self,X,y,W0,mm_steps=2,draws=1000,tune=1000,avoid_ZeroDivision=1e-16,
            **kwargs):
        """
        ``fit`` method for ``_LogisticMCPRegression_PyMc``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        W0 : array of shape (n_classes-1,n_features)
            Initial guess for MM algorithm.

        draws : int ; default 1000
            The number of samples to draw.

        tune : int ; default 1000
            Number of iterations to tune the step sizes, scalings etc. Tuning samples
            will be drawn in addition to the number specified in the `draws`,
            and will be discarded then.

        mm_steps : int ; default 2
            Number of iterations in MM algorithm.

        avoid_ZeroDivision : float ; default 1e-16
            A very small positive number to be substituted, to avoid ``ZeroDivisionError``

        **kwargs : other keyword arguments to ``WLasso_Logistic.fit(...)``.

        Returns
        -------
        self

        """
        self.estimator = WLasso_Logistic(n_classes=self.n_classes,
                                    avoid_ZeroDivision=avoid_ZeroDivision,
                                    random_state=self.random_state)
        ## MM algorithm .....
        while(mm_steps>0) :
            penalty_strength = self._rhd_MCP(W0)
            self.estimator.fit(X,y,penalty_strength,draws=draws,tune=tune,**kwargs)
            W0 = self.estimator.coef_
            mm_steps -= 1
        ## fitted coefficients .....
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        return self

    def predict_proba(self,X):
        return self.estimator.predict_proba(X)

    def score(self,X,y):
        return self.estimator.score(X,y)

#> ...............................................................




class MCP_importance(My_Template_FeatureImportance):
    """
        Feature selection based on minimax concave penalty (MCP).

        This penalty is not convex . It is concave , symmetric with respect to 0
        and increases with the magnitude of coefficients and
        eventually becomes constant after some threshold.

        We solve it with Majorize-Minimization(MM) algorithm using linear approximation.

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
        ..[1] Zhang, Cun-Hui. "Nearly unbiased variable selection under minimax concave penalty."
        (2010): 894-942.

        ..[3] Zou, Hui, and Runze Li. "One-step sparse estimates in nonconcave penalized likelihood models."
        Annals of statistics 36.4 (2008): 1509.

    """

    def __init__(self,random_state=None,*,
                 max_features=None,threshold=1e-10):
        super().__init__(random_state)
        self.max_features=max_features
        self.threshold=threshold


    def _initial_guess(self,X,y):
        """
        Initial value in MM algorithm.

        [ Default is `coef_` based on ``LogisticRegression().fit(X,y)``, override if necessary ]

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        array of shape (n_classes-1,n_features)

        [ last target class is considered as baseline for identifiability ]
        """
        guess = LogisticRegression().fit(X,y)
        return guess.coef_[:-1] - guess.coef_[-1]


    def _Estimator(self):
        """
        Which kind of method to be used for computation.

        [ override if necessary ]

        """
        return _LogisticMCPRegression_PyMc


    def fit(self,X,y,mcp_strength=[1.],mcp_concavity=[3.],*,
            cv_config={'cv':None,'n_jobs':None,'verbose':2},
            reduce_norm=1,
            **fit_params):
        """
        ``fit`` method for ``MCP_importance``

        Calling this method will start MM algorithm from beginning everytime. To resume an
        existing MM run, proceed as follows -

            >>> Model.fit(X,y,...) # fitting for the first time with required parameters
            >>> Model.estimator.fit(X,y,W0=Model.coef_) # resuming previous run
            >>> Model.update_importance() # update feature_importances_


        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : Series of shape (n_samples,)
            The target values.

        mcp_strength : list ; default [1.]
            Tuning parameter controlling the strength of MCP penalty.

        mcp_concavity : list ; default [3.]
            Tuning parameter controlling the concavity of MCP penalty.

        cv_config : dict of keyword arguments to ``GridSearchCV`` ; default ``{'cv':None,'n_jobs':None,'verbose':2}``
            Will be used when `mcp_strength` or `mcp_concavity` has to be determined by crossvalidation.

        reduce_norm : non-zero int, inf, -inf ; default 1
            Order of the norm used to compute `feature_importances_` in the case where the `coef_` of the
            underlying Logistic Regression is of dimension 2. By default 'l1'-norm is being used.

        **fit_params : other keyword arguments to ``_Estimator().fit(...)``.

        Returns
        -------
        self
            The fitted ``MCP_importance`` instance is returned.

        """
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        y = pd.Series(y,dtype='category')
        super().fit(X,y)
        ### assigning the estimator .....
        estimator = self._Estimator()
        estimator = estimator(self.n_classes_,
                              mcp_strength[0],mcp_concavity[0],
                              self.random_state)
        ### initial guess for MM algo .....
        W0 = self._initial_guess(X,y)
        ### fitting the model .....
        if len(mcp_strength)>1 or len(mcp_concavity)>1 :
            cv_config.update({'refit':True})
            self.gridsearch = GridSearchCV(estimator,
                                           param_grid={'strength':mcp_strength,'concavity':mcp_concavity},
                                           **cv_config)
            self.gridsearch.fit(X,y,W0=W0,**fit_params)
            self.estimator = self.gridsearch.best_estimator_
            self.best_penalty_ = self.gridsearch.best_params_
        else :
            estimator.fit(X,y,W0,**fit_params)
            self.estimator = estimator
            self.best_penalty_ = {'strength':mcp_strength[0],'concavity':mcp_concavity[0]}
        ### feature_importances .....
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        self.training_data = (X,y)
        self.reduce_norm = reduce_norm
        self.feature_importances_ = self._coef_to_importance(self.coef_,
                                                             reduce_norm,identifiability=True)
        return self


    def get_permutation_importances(self,test_data=(None,None),*,n_repeats=10):
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
        X_train,y_train = self.training_data
        X_test = X_train if (X_test is None) else pd.get_dummies(X_test,drop_first=True,dtype=int)
        y_test = y_train if (y_test is None) else pd.Categorical(y_test,
                                                                 categories=y_train.cat.categories)
        return super()._permutation_importance((X_test,y_test),n_repeats=n_repeats,
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
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        if imp_kind == 'permutation':
            self.get_permutation_importances(**kwargs_pimp)
        else :
            self.feature_importances_ = self._coef_to_importance(self.coef_,
                                                                 self.reduce_norm,identifiability=True)


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
