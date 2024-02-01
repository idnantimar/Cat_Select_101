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
from .. import My_Template_FeatureImportance,_Data_driven_Thresholding



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

        y : array-like of shape (n_samples,n_classes)
            The one-hot encoded target values.

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

        mcp_strength : list ; default [1.]
            Tuning parameter controlling the strength of MCP penalty.

        mcp_concavity : list ; default [3.]
            Tuning parameter controlling the concavity of MCP penalty.

        cv_config : dict of keyword arguments to ``GridSearchCV`` ; default ``{'cv':None,'verbose':2}``
            Will be used when `mcp_strength` or `mcp_concavity` has to be determined by crossvalidation.

        reduce_norm : non-zero int, inf, -inf ; default 1
            Order of the norm used to compute `feature_importances_` in the case where the `coef_` of the
            underlying Logistic Regression is of dimension 2. By default 'l1'-norm is being used.

        max_features : int ; default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

        threshold : float ; default 0
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise will be rejected.

        cumulative_score_cutoff : float in [0,1) ; default 0.01
            Computes data-driven 'threshold' for selecting those features that contributes to top
            100*(1-cut_off)% feature importances. Result is not valid when all features
            are unimportant.


        Attribures
        ----------
        best_penalty_ : the best value for regularization strength among `mcp_strength` or `mcp_concavity`,
        when ``len(mcp_strength)``>1 or ``len(mcp_convavity)``>1

        classes_ : array of shape (`n_classes_`,)
            A list of class labels known to the classifier.

        coef_ : array of shape (`n_classes_`-1, `n_features_in_`)
            Coefficient of the features in the decision function, considering last class as baseline.

        confusion_matrix_for_features_ : array of shape (`n_features_in_`, `n_features_in_`)
            ``confusion_matrix`` (`true_support`, `support_`)

        estimator : a fitted ``_Estimator(...)`` instance, having ``predict_proba`` and ``score`` method

        false_discoveries_ : array of shape (`n_features_in_`,)
            Boolean mask of false positives.

        false_negatives_ : array of shape (`n_features_in_`,)
            Boolean mask of false negatives.

        fdr_ : float
            1 - ``precision_score`` (`true_support`, `support_`)

        feature_importances_ : array of shape (`n_features_in_`,)
            Importances of features.

        feature_names_in_ : array of shape (`n_features_in_`,)
            Names of features seen during ``fit``.

        features_selected_ : array of shape (`n_features_selected_`,)
            Names of selected features.

        f1_score_for_features_ : float
            ``f1_score`` (`true_support`, `support_`)

        gridsearch : a fitted ``GridSearchCV(...)`` object, available when
        ``len(mcp_strength)``>1 or ``len(mcp_convavity)``>1

        intercept_ : array of shape (`n_classes_`-1,)
            Intercept added to the decision function, considering last class as baseline.

        minimum_model_size_ : int
            ``np.max`` (`ranking_` [ `true_support` ])

        n_classes_ : int
            Number of target classes.

        n_false_negatives_ : int
            Number of false negatives.

        n_features_in_ : int
            Number of features seen during ``fit``.

        n_features_selected_ : int
            Number of selected features.

        n_samples_ : int
            Number of observations seen during ``fit``.

        pcer_ : float
            ``np.mean`` (`false_discoveries_`)

        pfer_ : int
            ``np.sum`` (`false_discoveries_`)

        ranking_ : array of shape (`n_features_in_`,)
            The feature ranking, such that ``ranking_[i]`` corresponds to the
            i-th best feature, i=1,2,..., `n_features_in_`.

        support_ : array of shape (`n_features_in_`,)
            Boolean mask of selected features.

        threshold_ : float
            Cut-off in use, for selection/rejection.

        tpr_ : float
            ``recall_score`` (`true_support`, `support_`)

        true_support : array of shape (`n_features_in_`,)
            Boolean mask of active features in population, only available after
            ``get_error_rates`` method is called with true_imp.



        References
        ----------
        ..[1] Zhang, Cun-Hui. "Nearly unbiased variable selection under minimax concave penalty."
        (2010): 894-942.

        ..[3] Zou, Hui, and Runze Li. "One-step sparse estimates in nonconcave penalized likelihood models."
        Annals of statistics 36.4 (2008): 1509.

    """

    def __init__(self,random_state=None,*,mcp_strength=[1.],mcp_concavity=[3.],
                 cv_config={'cv':None,'verbose':2},
                 reduce_norm=1,
                 max_features=None,threshold=0,cumulative_score_cutoff=0.01):
        super().__init__(random_state)
        self.max_features=max_features
        self.threshold=threshold
        self.cumulative_score_cutoff = cumulative_score_cutoff
        self.mcp_strength = mcp_strength
        self.mcp_concavity = mcp_concavity
        self.cv_config = cv_config
        self.reduce_norm = reduce_norm


    def _initial_guess(self,X,y):
        """
        Initial value in MM algorithm.

        [ Default is `coef_` based on ``LogisticRegression(penalty='l2').fit(X,y)``, override if necessary ]

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
        guess = LogisticRegression(penalty='l2').fit(X,y)
        return guess.coef_[:-1] - guess.coef_[-1]


    def _Estimator(self):
        """
        Which kind of method to be used for computation.

        [ override if necessary ]

        """
        return _LogisticMCPRegression_PyMc


    def fit(self,X,y,**fit_params):
        """
        ``fit`` method for ``MCP_importance``

        Calling this method will start MM algorithm from beginning everytime. To resume an
        existing MM run, proceed as follows -

            >>> Model.fit(X,y,...) # fitting for the first time with required parameters
            >>> Model.estimator.fit(X,pd.get_dummies(y),W0=Model.coef_) # resuming previous run
            >>> Model.update_importance() # update feature_importances_


        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : Series of shape (n_samples,)
            The target values.

        **fit_params : other keyword arguments to ``_Estimator().fit(...)``.

        Returns
        -------
        self
            The fitted ``MCP_importance`` instance is returned.

        """
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        super().fit(X,y)
        ### assigning the estimator .....
        estimator = self._Estimator()
        estimator = estimator(self.n_classes_,
                              self.mcp_strength[0],self.mcp_concavity[0],
                              self.random_state)
        ### initial guess for MM algo .....
        W0 = self._initial_guess(X,y)
        ### fitting the model .....
        y = pd.get_dummies(y,dtype=int,drop_first=False)
        if len(self.mcp_strength)>1 or len(self.mcp_concavity)>1 :
            self.cv_config.update({'refit':True})
            self.gridsearch = GridSearchCV(estimator,
                                           param_grid={'strength':self.mcp_strength,'concavity':self.mcp_concavity},
                                           **self.cv_config)
            self.gridsearch.fit(X,y,W0=W0,**fit_params)
            self.estimator = self.gridsearch.best_estimator_
            self.best_penalty_ = self.gridsearch.best_params_
        else :
            estimator.fit(X,y,W0,**fit_params)
            self.estimator = estimator
        ### feature_importances .....
        self.coef_ = self.estimator.coef_
        self.intercept_ = self.estimator.intercept_
        self.feature_importances_ = self._coef_to_importance(self.coef_,
                                                             self.reduce_norm,identifiability=True)
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
        out = super()._permutation_importance((X_test.to_numpy(),pd.get_dummies(y_test,dtype=int,drop_first=False).to_numpy()),
                                               n_repeats=n_repeats,
                                               scoring=None)
        _Data_driven_Thresholding(self)
        return out


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
            _Data_driven_Thresholding(self)


    def transform(self,X):
        X = pd.get_dummies(X,drop_first=True,dtype=int)
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
               'PCER': per-comparison error rate,

               'FDR': false discovery rate,

               'PFER': per-family error rate,

               'TPR': true positive rate
            }

        """
        self.get_support()
        true_imp = np.array(true_imp)
        if (true_imp.dtype==bool) :
            self.true_support = true_imp
        else :
            self.true_support = (true_imp > self.threshold_)
        return super().get_error_rates(plot=plot)




#### ==========================================================================
