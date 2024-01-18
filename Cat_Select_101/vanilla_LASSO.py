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
from . import My_Template_FeatureImportance




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

        max_features : int ; default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

        threshold : float ; default 1e-10
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise will be rejected.

        Attribures
        ----------

    """
    Estimator_Type = {'penalty':'l1','solver':'saga'}

    def __init__(self,random_state=None,*,multi_class='multinomial',
                 max_features=None,threshold=1e-10):
        super().__init__(random_state)
        self.multi_class = multi_class
        self.threshold = threshold
        self.max_features = max_features


    def fit(self,X,y,Cs=np.logspace(-4,4,10),*,max_iter=1000,reduce_norm=1,**set_params):
        """
        ``fit`` method for ``vanillaLASSO_importance``.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : Series of shape (n_samples,)
            The target values.

        Cs : list of floats ; default ``np.logspace(-2,2,5)``
            The inverse of regularization strength.

        max_iter : int ; default 1000
            Maximum number of iterations of the optimization algorithm.

        reduce_norm : non-zero int, inf, -inf ; default 1
            Order of the norm used to compute `feature_importances_` in the case where the `coef_` of the
            underlying Logistic Regression is of dimension 2. By default 'l1'-norm is being used.

        **set_params : other keyword arguments to ``LogistiRegression()`` or ``LogistiRegressionCV()``

        Returns
        -------
        self
            The fitted ``vanillaLASSO_importance`` instance is returned.

        """
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        super().fit(X,y)
        ### fitting the model .....
        if len(Cs)>1 :
            estimator = LogisticRegressionCV(**self.Estimator_Type,
                                             multi_class=self.multi_class,
                                             Cs=Cs,
                                             random_state=self.random_state,max_iter=max_iter,
                                             **set_params)
            estimator.fit(X,y)
            self.C_ = estimator.C_
        else :
            estimator = LogisticRegression(**self.Estimator_Type,
                                           multi_class=self.multi_class,
                                           C=Cs[0],
                                           random_state=self.random_state,max_iter=max_iter,
                                           **set_params)
            estimator.fit(X,y)
            self.C_ = Cs[0]
        ### feature_importances .....
        self.estimator = estimator
        self.coef_ = estimator.coef_
        self.intercept_ = estimator.intercept_
        if self.multi_class=='multinomial' :
            self.feature_importances_ = self._coef_to_importance(self.coef_,reduce_norm,
                                                             identifiability=False)
        else :
            self.feature_importances_ = self._coef_to_importance(self.coef_,reduce_norm,
                                                             identifiability=True)
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
        return super()._permutation_importance((X_test,y_test),n_repeats=n_repeats,
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


    def plot(self,sort=True,savefig=None,**kwargs):
        ylabel = self.__class__.__name__+'|'+str(self.multi_class)
        kwargs.update({'ylabel':'vanillaLASSO_'+str(self.multi_class)})
        super().plot(sort=sort,savefig=savefig,**kwargs)


    def dump_to_file(self,file_path=None):
        if file_path is None :
            file_path = os.path.join(os.getcwd(),
                                     f"vanillaLASSO_{self.multi_class}-{datetime.now().strftime('%Y_%m_%d_%H%M%S%f')}.pkl")
        super().dump_to_file(file_path)





#### ==========================================================================
