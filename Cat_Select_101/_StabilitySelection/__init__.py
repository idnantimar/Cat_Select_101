"""
Created on Sun Dec 31 23:10:21 2023

Topic: Stabilty Selection is not any particular method,
        but a framework that can be applied on any base method for reduced variability on selected support.

Note : An implementation of Stability selection already available at https://github.com/scikit-learn-contrib/stability-selection .

(i) This one is just a bit customized as per our rest of the codes,
since we have already implemented ``get_support()`` in our `base_estimator`.

(ii) Instead of computing `support_` for a scalar tau, option for computing
support for multiple tau values at a time, to avoid redundant calculations.


@author: R.Nandi
"""




#### Complementary Pairs Stability Selection ===================================

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin,clone
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
from sklearn.metrics import precision_score,recall_score,f1_score
import warnings
import matplotlib.pyplot as plt
import os
from datetime import datetime
from ..vanilla_LASSO import vanillaLASSO_importance




class CPSS(TransformerMixin,BaseEstimator):
    """
        'Complementary Pairs Stability Selection' fits the `base_estimator` several times on
        two randomly chosen pairs of subsets of size [n_samples/2] each,
        for possibly different values of the regularization parameter. Variables that
        reliably get selected by the model in these iterations are
        considered to be stable variables.

        Parameters
        ----------
        base_estimator : object having ``fit`` and ``get_support`` method
            The base estimator used for stability selection.

        n_resamples : int ; default 50
            Number of times complementary pairs to be drawn

        verbose : bool ; default True
            Verbosity of the fitting procedure.

        lambda_name : str ; default None
            The name of the penalization parameter for the `base_estimator`.
            If None, default penalization in the initialization of base_estimator will
            be used.

        lambda_grid : list
            Grid of values of the penalization parameter to iterate over.

        kwargs_Parallel : dict of keyword arguments to ``joblib.Parallel`` ;
        default ``{'n_jobs':None}``

        random_state : int ; default None
            Seed for resampling. It will only ensure reproducability of the generated
            subsamples, and not of any randomness inside `base_estimator`.

        Attributes
        ----------
        feature_names_in_ : array of shape (`n_features_in_`,)
            Names of features seen during ``fit``.

        n_features_in_ : int
            Number of features seen during ``fit``.

        n_features_selected_ : int ( or array of int of shape (``len(tau)``,) )
            Number of selected features (for each value of `tau`).

        n_samples_ : int
            Number of observations seen during ``fit``.

        ranking_ : array of shape (`n_features_in_`,)
            The feature ranking, such that ``ranking_[i]`` corresponds to the
            i-th best feature, i=1,2,..., `n_features_in_`.

        stability_path_ : csr matrix of shape (``len(lambda_grid)``, n_features_in_)
            The regularization path in stability selection, when `lambda_grid` is provided.

        stability_scores_ : array of shape (`n_features_in_`,)
            The estimated selection probability for each features.

        support_ : array of shape (`n_features_in_`,) ( or shape (``len(tau)``,, `n_features_in_`) )
            Boolean mask of selected features (each row corresponds to one value of `tau`).

         tau : float (or list of floats) in (0,1)
             The cutoff value, any feature with stability score below this value will be rejected.

        true_support : array of shape (`n_features_in_`,)
            Boolean mask of active features in population, only available after
            ``get_error_rates`` method is called with true_imp.

        References
        ----------
        .. [1] Shah, R.D. and Samworth, R.J., 2013. Variable selection with
               error control: another look at stability selection. Journal
               of the Royal Statistical Society: Series B (Statistical Methodology),
                75(1), pp.55-80.

        .. [2] Meinshausen, N. and Buhlmann, P., 2010. Stability selection.
               Journal of the Royal Statistical Society: Series B
               (Statistical Methodology), 72(4), pp.417-473.

    """
    def __init__(self,base_estimator,
                 n_resamples=20,*,verbose=True,
                 lambda_name=None,lambda_grid=None,
                 kwargs_Parallel={'n_jobs':None},random_state=None):
        self.base_estimator = base_estimator
        self.n_resamples = n_resamples
        self.verbose = verbose
        self.lambda_name = lambda_name
        self.lambda_grid = lambda_grid
        self.kwargs_Parallel = kwargs_Parallel
        self.random_state = random_state


    def _subsamples(self,n,rng):
        """
        Generate the complementary pairs indices.

        [ for internal use only ]

        """
        all_samples = range(n)
        n_ = (n//2)*2
        subsamples = np.empty((2*self.n_resamples,n//2),dtype=int)
        for i in range(self.n_resamples):
            current = rng.permutation(all_samples)[:n_]
            subsamples[2*i:2*(i+1)] = current.reshape((2,-1))
        return subsamples


    def fit(self,X,y,**fit_params):
        """
        ``fit`` method for Complementary Pairs Stability Selection.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : Series of shape (n_samples,)
            The target values.

        **fit_params : other keyword arguments to the ``fit`` method of `base_estimator`.

        Returns
        -------
        self
            The fitted CPSS instance is returned.

        """
        X,y = pd.DataFrame(X),pd.Series(y)
        self.n_samples_,p = X.shape
        ### Generating complementary pairs .....
        subsamples = self._subsamples(n=self.n_samples_,
                                      rng=np.random.default_rng(self.random_state))
        ### Feature Selection/Rejection for each subsample of size [n/2] .....
        def for_a_subsample(index,estimator=self.base_estimator):
            estimator.fit(X.iloc[index],y.iloc[index],**fit_params)
            return estimator.get_support()
        ### The main computation chunk for Stability Scores .....
        if self.lambda_name is None :
            selected_ = Parallel(**self.kwargs_Parallel)(
                delayed(for_a_subsample)(row) for row in subsamples)
            self.stability_scores_ = np.mean(selected_,axis=0)
        else :
            self.stability_path_ = []
            n_lambdas = len(self.lambda_grid)
            for idx,lambda_value in enumerate(self.lambda_grid) :
                if self.verbose :
                    print(f"Fitting base_estimator for lambda = {lambda_value} ============ case : [{idx+1} / {n_lambdas}]")
                estimator = clone(self.base_estimator).set_params(**{self.lambda_name:lambda_value})
                on_each_subsample = lambda index : for_a_subsample(index,estimator)
                selected_ = Parallel(**self.kwargs_Parallel)(
                    delayed(on_each_subsample)(row) for row in subsamples)
                self.stability_path_ += [np.mean(selected_,axis=0)]
            self.stability_scores_ = np.max(self.stability_path_,axis=0)
            self.stability_path_ = csr_matrix(self.stability_path_)
        ### return values .....
        self.n_features_in_ = len(self.stability_scores_)
        if p != self.n_features_in_ :
            warnings.warn(f"""``CPSS()`` fitted successfully, but X.shape[1]={p} is not same as
                              `n_features_in_`={self.n_features_in_}, ``transform()`` can not be called directly.
                              e.g. - if `base_estimator` internally use ``pd.get_dummies(X)`` during fit, use
                                  >>> selector = CPSS(base_estimator)
                                  >>> selector.fit(X,y) # you are currently here, successfully
                                  >>> selector.transform(pd.get_dummies(X))
                          """)
        else : self.feature_names_in_ = X.columns
        self.ranking_ = rankdata(-self.stability_scores_,method='average')
        return self


    def get_support(self,tau=0.6):
        """
        Get a boolean mask of the features selected.

        Parameters
        ----------
        tau : float or list of floats in (0,1) ; default 0.6
            The cutoff in Stability Selection. Any feature with stability score below
            this cutoff will be rejected, otherwise selected.

        Returns
        -------
        array of shape (`n_features_in_`,) when tau is a float

        array of shape (len(tau),`n_features_in_`) when tau is a list,
        each row corresponds to one element of tau.

        """
        self.tau = np.array(tau)
        self.support_ = (self.stability_scores_ >= self.tau.reshape((-1,1)))
        if self.tau.shape==() : self.support_ = self.support_.ravel()
        self.n_features_selected_ = self.support_.sum(axis=-1,keepdims=False)
        return self.support_


    def transform(self,X,tau=0.6):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, `n_features_in_`)
            The input samples to be transformed.

        tau : float in (0,1) ; default 0.6
            The cutoff in Stability Selection. Any feature with stability score below
            this cutoff will be rejected, otherwise selected.

        Returns
        -------
        DataFrame of shape (n_samples, n_selected_features)
            The input samples with only the selected features.

        """
        return X.iloc[:,self.get_support(tau)]


    def fit_transform(self,X,fit_params):
        """
        Ignored. Since `n_features_in_` can be different from X.shape[1] .
        Use ``fit`` and ``transform`` separately.

        """
        pass


    def plot(self,title='Stability Path',savefig=None):
        """
        Stability Path for the features, for given `lambda_grid`.

        [ savefig=True will save the plot inside a folder PLOTs at the current working directory.
        The plot will be saved as self.__class__.__name__-datetime.now().strftime('%Y_%m_%d_%H%M%S%f').png ]

        """
        if (self.lambda_name is not None) :
            cmap = plt.colormaps.get_cmap('tab10')
            names = getattr(self,'feature_names_in_',['X_'+str(j) for j in range(self.n_features_in_)])
            regularization = np.ravel(self.lambda_grid)
            stability_path = self.stability_path_.toarray()[regularization.argsort()]
            regularization.sort()
            for idx in range(self.n_features_in_) :
                plt.plot(regularization,stability_path[:,idx],
                         label=names[idx],color=cmap(idx))
            plt.xlabel('regularization')
            plt.ylabel('score')
            plt.title(title)
            plt.legend()
            plt.ylim(0, 1)
            plt.xticks(rotation=30)
            if savefig is not None :
                if savefig==True :
                    os.makedirs('PLOTs',exist_ok=True)
                    savefig = 'PLOTs'
                plt.savefig(os.path.join(savefig,
                                         f"{self.__class__.__name__}-{datetime.now().strftime('%Y_%m_%d_%H%M%S%f')}.png"))
            plt.show()
        else : print("`lambda_name` was not provided during initialization.")


    def _error_rates(self,current_support,true_support):
        """
        This function computes attributes `pfer_`, `pcer_`, `fdr_`,
        `tpr_`, `f1_score_for_features_` .

        [ for internal use only ]

        """
        compare_truth = (np.array(true_support,dtype=int)-current_support)
        false_discoveries_ = (compare_truth == -1)
                ## if a feature is True in 'current_support' and False in 'true_support'
                 ## it is a false-discovery or false +ve
        pfer_ = false_discoveries_.sum()
        pcer_ = pfer_/self.n_features_in_
        fdr_ = 1 - precision_score(y_true=true_support,y_pred=current_support,
                                         zero_division=1.0)
                ## lower 'fdr_' is favourable
        tpr_ = recall_score(y_true=true_support,y_pred=current_support,
                                 zero_division=np.nan)
                ## higher 'tpr_' is favourable
        f1_score_for_features_ = f1_score(y_true=true_support,y_pred=current_support,
                                               zero_division=np.nan)
                ## this confusion matrix or f1 score corresponds to the labelling of
                 ## null/non-null features, not corresponds to the labelling of target(y) classes
        return {'PCER':pcer_,
                'FDR':fdr_,
                'PFER':pfer_,
                'TPR':tpr_,
                'F1score_for_features':f1_score_for_features_}


    def get_error_rates(self,true_support,tau=0.6):
        """
        Computes various error-rates when true support of the model is known.

        Parameters
        ----------
        true_support : boolean array of shape (`n_features_in_`,)
            True implies the feature is important in true model, null feature otherwise.

        tau : float or list of floats in (0,1) ; default 0.6
            The cutoff in Stability Selection. Any feature with stability score below
            this cutoff will be rejected, otherwise selected.

        Returns
        -------
        DataFrame
            Where each row corresponds to one value of tau.

        """
        tau_values = np.array(tau).reshape((-1,))
        current_supports = self.get_support(tau).reshape((len(tau_values),-1))[tau_values.argsort()]
        tau_values.sort()
        self.true_support = np.array(true_support)
        error_rates = []
        for current_support in current_supports :
            error_rates += [self._error_rates(current_support,self.true_support)]
        return pd.DataFrame(error_rates,index=tau_values)


    def dump_to_file(self,file_path=None):
        """
        Save this current instance to a specified file location.

        It can be loaded later as follows-

        >>> with open('your_saved_instance.pkl', 'rb') as file:
        ...:     loaded_instance = joblib.load(file)
        >>> loaded_instance

        Parameters
        ----------
        file_path : "path/to/your_file" with a trailing .pkl
            The default is None, which will save the file at the current working directory as
            self.__class__.__name__-datetime.now().strftime('%Y_%m_%d_%H%M%S%f').pkl

        """
        if file_path is None :
            file_path = os.path.join(os.getcwd(),
                                     f"{self.__class__.__name__}-{datetime.now().strftime('%Y_%m_%d_%H%M%S%f')}.pkl")
        if os.path.exists(file_path):
            user_input = input(f"File '{file_path}' already exists.\n Do you want to overwrite it? (YES/no): ").upper()
            if user_input != 'YES':
                print("Continue with a different filename.")
                return
        with open(file_path, 'wb') as file:
            joblib.dump(self, file)
        print(f"Instance saved as {file_path} successfully...")





#### ==========================================================================
