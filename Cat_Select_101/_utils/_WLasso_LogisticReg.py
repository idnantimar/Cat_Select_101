"""
Created on Wed Jan  3 18:09:03 2024

Topic: Weighted Lasso ('l1'-penalty, but different weights attached to different coefficients) implementation.
       For multi-class classification case.

NOTE : In the case of linear regression (or binary classification), when `coef_` is of dimension
(`n_features_in_`) Weighed LASSO can be solved with existing ``ElasticNet(l1_ratio=1)`` or
(``LogisticRegression(penalty='l1')``) in ``sklearn``, by just reweighting the columns of ``X`` accordingly. For
Multinomial Logistic Regression setup that is not possible, since
`coef_` is of shape (`n_classes_`,`n_features_in_`), coefficients in same column but different
rows can have different weightings.

So here we try a simple implementation of the problem from scratch using ``PyMC``.

Key Idea : LASSO corresponds to MAP estimate using Laplace(0,sigma) prior. If we use different
sigma for different features, we will have Weighted LASSO.


@author: R.Nandi

COMMENT for future use: Not stable when weights are near 0. Subject to improvement.
"""


#### Logistic Regression + Weighted LASSO penalty =============================

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy.stats import multinomial




class WLasso_Logistic(BaseEstimator):
    """
    Key Idea : LASSO corresponds to MAP estimate using Laplace(0,sigma) prior. If we use different
    sigma for different features, we will have Weighted LASSO.

    [ Here weight is on features(columns of X), not on samples(rows of X) ]

    Parameters
    ----------
    n_classes : int
        The number of target classes in consideration.

    avoid_ZeroDivision : float ; default 1e-16
        A very small positive number to be substituted, to avoid ``ZeroDivisionError``

    random_state : int ; default None
        Seed for reproducible results across multiple function calls.

    """
    def __init__(self,n_classes,avoid_ZeroDivision=1e-16,random_state=None):
        self.n_classes = n_classes
        self.avoid_ZeroDivision = avoid_ZeroDivision
        self.random_state = random_state



    def fit(self,X,y,penalty_strength=1e-1,*,
            draws=1000,tune=1000,
            kwargs_Model={},
            kwargs_Sample={'progressbar':True,'compute_convergence_checks':True,'keep_warning_stat':True},
            kwargs_MAP={'progressbar':False}):
        """
        ``fit`` method for Weighted Lasso.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,n_classes)
            The one-hot encoded target values.

        penalty_strength : array of shape (n_classes-1,n_features) or scaler ; default 0.1
            The weights corresponding to each 'l1'-penalties. Each entry must be non-negative.
            More important features should have smaller penalty_strength.

            (For identifiability we will consider the last response category as baseline)

        draws : int ; default 1000
            The number of samples to draw.

        tune : int ; default 1000
            Number of iterations to tune the step sizes, scalings etc. Tuning samples
            will be drawn in addition to the number specified in the `draws`,
            and will be discarded then.

        kwargs_Model,kwargs_Sample,kwargs_MAP : dict of other keyword arguments to ``pymc.Model(...)``,
        ``pymc.sample(...)`` and ``pymc.find_MAP(...)`` respectively.


        Returns
        -------
        self

        Attributes
        ----------
        estimator : fitted ``pymc.Model(...)``

        trace : object that contains the posterior samples.

        coef_ : array of shape (n_classes-1,n_features)
            The estimated coefficients.

        intercept_ : array of shape (n_classes-1,)
            The estimated intercepts.

        """
        with pm.Model(**kwargs_Model) as Multinomial_Logistic:
            ## observed data ....
            n_features = X.shape[1]
            X = pm.ConstantData('features',value=X)
            y = pm.ConstantData('target',value=y)
            ## Laplace priors for weights ....
            w0 = np.where(penalty_strength>self.avoid_ZeroDivision,
                          penalty_strength,self.avoid_ZeroDivision)
            W = pm.Laplace('coef',mu=0,b=np.ones((self.n_classes-1,n_features))/w0)
            ## virtually flat priors for bias ....
            b = pm.Normal('intercept',mu=np.zeros((self.n_classes-1,)),sigma=1e+2)
            ## likelihood function ....
            logits = pm.math.dot(X,W.T)+b
            e_logits = pm.math.exp(logits)
            e_logits_sum = 1 + pm.math.sum(e_logits,axis=1).reshape((-1,1))
            probs = pm.math.concatenate([e_logits/e_logits_sum,1/e_logits_sum],axis=1)
            LIKELIHOOD = pm.Multinomial("WLasso_Logistic",n=1,p=probs,observed=y)

        ### running MCMC .....
        with Multinomial_Logistic:
            trace = pm.sample(random_seed=self.random_state,discard_tuned_samples=True,
                              draws=draws,tune=tune,
                              **kwargs_Sample)

        ### fitted coef & intercept .....
        self.estimator = Multinomial_Logistic
        self.trace = trace
        maximum_a_posteriori = pm.find_MAP(model=Multinomial_Logistic,**kwargs_MAP)
        self.coef_ = maximum_a_posteriori['coef']
        self.intercept_ = maximum_a_posteriori['intercept']

        return self



    def diagnostics(self,
                    plots=[az.plot_trace,az.plot_posterior,az.plot_forest]):
        """
        Some basic visual scrutiny check of the fitted Model.

        Further analysis can be done using `estimator` and `trace`.

        """
        check_is_fitted(self)
        for plot in plots :
            plot(self.trace)



    def predict_proba(self,X):
        """
        ``predict_proba`` method for Weighted Lasso.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        array of shape (n_samples,n_classes)
            The probability estimates.

        """
        check_is_fitted(self)
        logits = np.matmul(np.array(X),self.coef_.T) + self.intercept_
        logits -= logits.max(axis=1,keepdims=True)
        e_logits = np.exp(logits)
        e_logits_sum = 1 + e_logits.sum(axis=1,keepdims=True)
        return np.concatenate([e_logits/e_logits_sum,1/e_logits_sum],axis=1)



    def score(self,X,y):
        """
        ``score`` method for Weighted Lasso. Computes the fitted average log-likelihood.
        Higher the value, better the fit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,n_classes)
            The one-hot encoded target values.

        Returns
        -------
        float

        """
        probs = self.predict_proba(X)
        l = multinomial.logpmf(y,n=1,p=probs)
        return np.mean(l)



#### ==========================================================================
