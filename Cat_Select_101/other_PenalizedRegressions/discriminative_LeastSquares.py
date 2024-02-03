"""
Created on Thu Jan 18 00:52:15 2024

Topic: Feature Importance Based on Discriminative Least Squares Regression.

@author: R.Nandi
"""



#### Discriminative Least Squares Regression for feature selection ============

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from .. import My_Template_FeatureImportance,_Data_driven_Thresholding
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
        regularization : list ; default [0.1]
            The strength of regularization.

        cv_config : dict of keyword arguments to ``GridSearchCV`` ; default ``{'cv':None,'verbose':2}``
            Will be used when `regularization` has to be determined by crossvalidation.

        max_iter : int ; default 30
            The maximum number of iterations.

        reduce_norm : non-zero int, inf, -inf ; default 2
            Order of the norm used to compute `feature_importances_` in the case where the `coef_` of the
            underlying model is of dimension 2. By default 'l2'-norm is being used.

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

        u : float ; default 1e+4
            A large positive number (virtually +ve infinity).

        tol : float ; default 1e-4
            The tolerence for convergence criterion.


        Attribures
        ----------
        best_penalty_ : the best value for regularization strength among `regularization`, when ``len(regularization)``>1

        category_specific_importances_ : array of shape (`n_classes_`, `n_features_in_`)
            Feature importances specific to each target class.

        classes_ : array of shape (`n_classes_`,)
            A list of class labels known to the classifier.

        coef_ : array of shape (`n_classes_`, `n_features_in_`)
            Coefficient of the features in the decision function.

        confusion_matrix_for_features_ : array of shape (`n_features_in_`, `n_features_in_`)
            ``confusion_matrix`` (`true_support`, `support_`)

        estimator : a fitted ``d_LSR(...)`` instance, having ``predict`` and ``score`` method

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

        gridsearch : a fitted ``GridSearchCV(...)`` object, available when ``len(regularization)``>1

        intercept_ : array of shape (`n_classes_`,)
            Intercept added to the decision function.

        minimum_model_size_ : int
            ``np.max`` (`ranking_` [ `true_support` ])

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
        ..[1] Xiang, Shiming, et al. "Discriminative least squares regression for multiclass
        classification and feature selection." IEEE transactions on neural networks and learning
        systems 23.11 (2012): 1738-1754.


    """
    def __init__(self,*,regularization=[0.1],cv_config={'cv':None,'verbose':2},
                 max_iter=30,reduce_norm=2,
                 max_features=None,threshold=0,cumulative_score_cutoff=0.05,
                 u=1e+4,tol=1e-4,):
        super().__init__()
        self.regularization = regularization
        self.cv_config = cv_config
        self.max_iter = max_iter
        self.reduce_norm = reduce_norm
        self.max_features = max_features
        self.threshold = threshold
        self.cumulative_score_cutoff = cumulative_score_cutoff
        self.u = u
        self.tol = tol


    def fit(self,X,y,*,initial_guess=(None,None)):
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

        initial_guess : tuple (W0,b0)
            Where,
            W0 is array of shape (n_features,n_classes)
            and b0 is array of shape (1,n_classes)

            When None, will be initialized at ``np.zeros()``.

        Returns
        -------
        self
            The fitted ``dLS_impotance`` instance is returned.

        """

        X = pd.get_dummies(X,dtype=float,drop_first=True)
        super().fit(X,y)
        y = pd.get_dummies(y,dtype=float,drop_first=False)
        ### assigning the estimator .....
        estimator = d_LSR(self.regularization[0],self.u)
        W0,t0 = initial_guess
        if W0 is None : W0 = np.zeros((self.n_features_in_,self.n_classes_),dtype=float)
        if t0 is None : t0 = np.zeros((1,self.n_classes_),dtype=float)
        ### fitting the Model .....
        fit_params = {'W0':W0,'t0':t0,'max_iter':self.max_iter,'tol':self.tol}
        if len(self.regularization)>1 :
            cv_config.update({'refit':True})
            self.gridsearch = GridSearchCV(estimator,
                                           param_grid={'regularization':self.regularization},
                                           **self.cv_config)
            self.gridsearch.fit(X.to_numpy(),y.to_numpy(),**fit_params)
            self.estimator = self.gridsearch.best_estimator_
            self.best_penalty_ = self.gridsearch.best_params_['regularization']
        else :
            estimator.fit(X.to_numpy(),y.to_numpy(),**fit_params)
            self.estimator = estimator
        ### feature_importances .....
        self.coef_ = self.estimator.coef_.T
        self.intercept_ = self.estimator.intercept_.ravel()
        self.feature_importances_ = self._coef_to_importance(self.coef_,
                                                             self.reduce_norm,identifiability=True)
        self.category_specific_importances_ = np.abs(self.coef_)**self.reduce_norm
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
        self.coef_ = self.estimator.coef_.T
        self.intercept_ = self.estimator.intercept_.ravel()
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
