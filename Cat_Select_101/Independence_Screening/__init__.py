"""
Created on Thu Jan 11 23:46:38 2024

Topic: Usually variable selection methods rely on the coefficients in some parametric model of Y|X=x.
       But for Categorical Response case, Nonparametric tests for homogeneity can be applied for variable selection.
       This methods can specially work well for mis-specified models.

@author: R.Nandi
"""



#### Model-Free approach to variable selection (for Continuous Predictors) ====

import numpy as np
import pandas as pd
from .. import My_Template_FeatureImportance



class SIS_importance(My_Template_FeatureImportance):
    """
        Key Idea : In the setup of categorical response Y, a feature Xj is unimportant
        if Xj|Y=y are very similar for all categories y.

        Note : This method only uses marginal utilities, ignoring joint covariance structure
        of the features.

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

        Attribures
        ----------

        References
        ----------
        ..[1] Cui, Hengjian, Runze Li, and Wei Zhong. "Model-free feature screening
        for ultrahigh dimensional discriminant analysis."
        Journal of the American Statistical Association 110.510 (2015): 630-641.

    """

    _coef_to_importance = None
    _permutation_importance = None
    def __init__(self,random_state=None,*,
                 max_features=None,threshold=1e-10):
        super().__init__(random_state)
        self.threshold = threshold
        self.max_features = max_features


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
        class_probs = u.mean(axis=1,keepdims=True)
        ### computation for each feature .....
        u = (u/class_probs) - 1
                ## this will be useful for calculating [F(X|Y) - F(X)]
        def for_jColumn(Xj):
            out = 0
            for x in Xj:
                v = np.mean((Xj<=x)*u,axis=1,keepdims=True)
                    ## this quantity is [F(X|Y) - F(X)]
                out += np.sum(class_probs*(v**2),axis=None)
            return out/self.n_samples_
        ### iterating over the columns .....
        X = X.to_numpy()
        self.feature_importances_ = np.apply_along_axis(for_jColumn,
                                                        axis=0,arr=X)
        return self


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
            self.true_support = (self.true_coef > self.threshold_)
        return super().get_error_rates(plot=plot)




#### ==========================================================================
##
###
####
###
##
#### For Categorical Predictors ===============================================


##> ............................................................
from scipy.stats import entropy


def _testHomogeneity_Categorical(x,y,seed=None,*,n_resamples=200):
    ## returns the p-value for testing H0: "x & y have same pmf" vs H1: not H0
    ## based on permutation test of Jensen-Shannon divergence, a right-tailed test
    n1,n2 = len(x),len(y)
    pooled_data = pd.concat([x,y])
    def JS_Div(a,b):
        a = pd.get_dummies(a).mean(axis=0)
        b = pd.get_dummies(b).mean(axis=0)
        m = (a+b)/2
        return (entropy(a,m)+entropy(b,m))
        ## we use 2*(JS divergence) as test statistic, >=0
        ## larger the value, stronger the evidence againt H0
    T_obs = JS_Div(x,y)
    generator = np.random.default_rng(seed)
    def T_i(i):
        resampled_data = pd.Series(generator.permutation(pooled_data),dtype='category')
        x_,y_ = resampled_data[:n1],resampled_data[n1:]
        return JS_Div(x_,y_)
    T_simulated = list(map(T_i,range(n_resamples)))
    return (np.array(T_simulated)>=T_obs).mean()

#> .............................................................


from itertools import combinations


class SIScat_importance(My_Template_FeatureImportance):
    """
        Key Idea : In the setup of categorical response Y, a feature Xj is unimportant
        if Xj|Y=y are very similar for all categories y.

        Note : This method only uses marginal utilities, ignoring joint covariance structure
        of the features.

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

        Attribures
        ----------

        References
        ----------
        ..[1] Sen, Sweata, Damitri Kundu, and Kiranmoy Das.
        "Variable selection for categorical response: A comparative study."
        Computational Statistics 38.2 (2023): 809-826.

    """

    _coef_to_importance = None
    _permutation_importance = None
    def __init__(self,random_state=None,*,
                 max_features=None,threshold=1e-10):
        super().__init__(random_state)
        self.threshold = threshold
        self.max_features = max_features

    def fit(self,X,y,alpha=0.05,
            testing_rule=_testHomogeneity_Categorical,**kwargs):
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

        alpha : float in (0,1) ; default 0.05
            The level of significance for testing each feature.

            [ Note: Due to Bonferroni correction, for multiple(>2) categories may have very small power ]

        testing_rule : a function of type ``test(group1,group2,seed,**kwargs)->pvalue`` ;
        default option is a permutation test based on Jensen-Shannon divergence.

        **kwargs : other keyword arguments to `testing_rule`.

        Returns
        -------
        self
            The fitted ``SIScat_importance`` instance is returned.
        """
        super().fit(X,y)
        y = pd.get_dummies(y,dtype=bool).to_numpy()
        X = X.astype('category')
        ### bonferroni correction .....
        k = self.n_classes_
        all_pairs = list(combinations(range(k),2))
        alpha /= (k*(k-1)/2)
        ### computation for each feature .....
        def for_jColumn(Xj):
            p_val = []
            for Group1,Group2 in all_pairs :
                Group1 = Xj[y[:,Group1]]
                Group2 = Xj[y[:,Group2]]
                p_val += [testing_rule(Group1,Group2,self.random_state,**kwargs)]
            return p_val
        ### iterating over the columns .....
        pvalues = X.apply(for_jColumn,axis=0).to_numpy()
        ### higher criticism approach .....

        return self


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
            self.true_support = (self.true_coef > self.threshold_)
        return super().get_error_rates(plot=plot)



#### ==========================================================================
