"""
Created on Sat Dec 23 10:13:35 2023

Topic: Penalized Regression based feature selection (for categorical response).

@author: R.Nandi
"""



#### Logistic Regression + Custom Penalties ===================================

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from .._utils._custom_LogisticReg import _custom_LogisticReg
from .. import My_Template_FeatureImportance,_Data_driven_Thresholding




##> ...................................................................

class custom_penalty(keras.regularizers.Regularizer):
    """
        A format for custom penalty to be applied. Default is 0-penalty.

        [ Override the ``__call__`` method accordingly, that will return the total penalty upon `coef` ]

        The ``__init__`` method must be intact in terms of a single tuning parameter `penalty_param`,
        which can be any thing like- scalar, array, list etc.
        e.g. - if we need two tuning parameters `alpha`, `beta`, the ``__call__`` method should look like

            >>>     def __call__(self,coef):
            ...:        alpha,beta = self.penalty_param
            ...:        ## rest of the code, in terms of `alpha`, `beta`

        And then the class should be instantiated as -

            >>> instance = custom_penalty((`alpha0`,`beta0`))

    """
    def __init__(self,penalty_param,dtype):
        self.penalty_param = penalty_param
        self.dtype = dtype
    def __call__(self,coef):
        return tf.constant(0,dtype=self.dtype)
    def get_config(self):
        return {'penalty_param':self.penalty_param}


#> ...................................................................


#### Logistic Regression + any penalty ========================================


class penalizedLOGISTIC_importance_tf(My_Template_FeatureImportance):
    """
        Feature selection using Logistic Regression with any custom penalty , implemented using ``tensorflow`` and ``sklearn``.

        Parameters
        ----------
        random_state : int ; default None
            Seed for reproducible results across multiple function calls.

        penalty_params : list
            Possible values of `penalty_param` tuning the shape of custom penalty function.
            The best one will be chosen by ``GridSearchCV``. For a list of length 1, it will be used as it is.

        dtype : any ``tf`` float dtype ; default `tf.float64`
            The dtype used in the underlying ``tensorflow`` neural network.

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

        cumulative_score_cutoff : float in [0,1) ; default 0.05
            Computes data-driven 'threshold' for selecting those features that contributes to top
            100*(1-cut_off)% feature importances. Result is not valid when all features
            are unimportant.

        compile_configuration : dict of keyword arguments for model compilation ; default {``'optimizer'``:'adam', ``'scoring'``:['accuracy']}
            Must include ``optimizer`` and the ``metrics`` will be used to evaluate ``scoring`` in ``GridSearchCV``.

        cv_configuration : dict of keyword arguments for ``GridSearchCV`` ; default ``{'cv':None,'verbose':2}``

    """

    def __init__(self,random_state=None,*,penalty_params=[0],dtype=tf.float64,
                 reduce_norm=1,
                 max_features=None,threshold=0,cumulative_score_cutoff=0.05,
                 compile_configuration={'optimizer':'adam','metrics':['accuracy']},
                 cv_configuration={'cv':None,'verbose':2}):
        super().__init__(random_state)
        self.penalty_params = penalty_params
        self.dtype = dtype
        self.reduce_norm = reduce_norm
        self.threshold = threshold
        self.max_features = max_features
        self.cumulative_score_cutoff = cumulative_score_cutoff
        self.compile_configuration = compile_configuration
        self.cv_configuration = cv_configuration


    def penalty(self):
        """
        The custom penalty function class implementation. Default is no penalty.

        [ Override it accordingly ]

        The ``__init__`` method must have a single parameter tuning the shape of the penalty function.
        There must be a ``__call__`` method.

        Returns
        -------
        The custom penalty function class (not an instance or a callable)
        """
        return custom_penalty


    def _Estimator(self):
        """
        Implementation of the underlying estimator.

        Returns
        -------
        A ``BaseEstimator`` subclass, with ``fit`` and ``score`` method.

        """
        return _custom_LogisticReg


    def _initial_guess(self):
        """
        Any classifier with a `coef_` and an `intercept_` attribute,
        those will be used as ``kernel_initializer`` and ``bias_initializer``
        of the underlying neural network to leverage the training.

        [ Override it accordingly ]

        Default choice is ``LogisticRegression(penalty='l2',multi_class='multinomial',solver='lbfgs',tol=1e-2)``
        """
        return LogisticRegression(penalty='l2',multi_class='multinomial',solver='lbfgs',tol=1e-2)


    def fit(self,X,y,*,epochs=100,**fit_params):
        """
        ``fit`` method for the Logistic Regression with custom penalty.

        Note : each time calling ``fit`` will compile the model based on the shape of the data, so previous record will be lost.
        To resume a previous training run proceed as follows

            >>> Model.fit(X,y,...) # fitting for the first time with required parameters
            >>> Model.estimator.fit(X,pd.get_dummies(y)) # resuming previous run
            >>> Model.update_importance() # update feature_importances_

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : Series of shape (n_samples,)
            The target values.

        epochs : int ; default 100
            The number of iterations the model will be trained.

        **fit_params : other keyword arguments, like- callback,validation_split,verbose etc.
        for ``fit`` method of underlying neural network.

        Returns
        -------
        self
            The fitted instance is returned.

        """
        X = pd.get_dummies(X,drop_first=True,dtype=self.dtype.as_numpy_dtype)
        super().fit(X,y)
        initial_guess = self._initial_guess()
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        initial_guess.fit(X, y)
        warnings.filterwarnings("default", category=ConvergenceWarning)
        y = pd.get_dummies(y,drop_first=False,dtype=int)

        fit_configuration = {'weights0':initial_guess.coef_.T,'bias0':initial_guess.intercept_,
                             'epochs':epochs,
                             'shuffle':(self.random_state is None)}
        fit_configuration.update(**fit_params)

        Estimator = self._Estimator()
        estimator = Estimator(n_features_in=self.n_features_in_,
                                 n_classes=self.n_classes_,
                                 penalty_class=self.penalty(),
                                 compile_configuration=self.compile_configuration,
                                 penalty_param=self.penalty_params[0],
                                 random_state=self.random_state,
                                 dtype=self.dtype)
                    ## this estimator is an _custom_LogisticReg() instance, imported from _utils

        if len(self.penalty_params)>1 :
            self.cv_configuration.update({'refit':True})
            self.gridsearch = GridSearchCV(estimator,param_grid={'penalty_param':self.penalty_params},
                                           **self.cv_configuration)
            self.gridsearch.fit(X,y,**fit_configuration)
            self.estimator = self.gridsearch.best_estimator_.nn
                    ## this self.estimator is the best fitted neural network
            self.best_penalty_ = self.gridsearch.best_params_['penalty_param']
        else :
            X,y = tf.convert_to_tensor(X),tf.convert_to_tensor(y)
            estimator.fit(X,y,**fit_configuration)
            self.estimator = estimator.nn
                    ## skip the GridSearch when only one possible penalty_param is given

        self.coef_ = (self.estimator.weights[0]).numpy().T
        self.intercept_ = (self.estimator.weights[1]).numpy().T
        self.feature_importances_ = self._coef_to_importance(self.coef_,
                                                             self.reduce_norm,identifiability=False)
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
        X_test = tf.convert_to_tensor(pd.get_dummies(X_test,drop_first=True,dtype=self.dtype.as_numpy_dtype))
        y_test = pd.Categorical(y_test,categories=self.classes_)
        y_test = tf.convert_to_tensor(pd.get_dummies(y_test,drop_first=False,dtype=int))
        out = super()._permutation_importance((X_test,y_test),n_repeats=n_repeats,
                                               scoring = lambda model,X,y : model.evaluate(X,y,verbose=0)[1])
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
        self.coef_ = (self.estimator.weights[0]).numpy().T
        self.intercept_ = (self.estimator.weights[1]).numpy().T
        if imp_kind == 'permutation':
            self.get_permutation_importances(**kwargs_pimp)
        else :
            self.feature_importances_ = self._coef_to_importance(self.coef_,
                                                                 self.reduce_norm,identifiability=False)
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
