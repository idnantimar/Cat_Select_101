"""
Created on Thu Feb  1 13:32:51 2024

Topic: Sometimes it is difficult to determine the raw cutoff for variable selection.
       Instead we can use some synthetic noise features to check how are their importances.
       Then all those original features having more importance than noise features will be selected.

@author: R.Nandi
"""



#### BCT ======================================================================

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator,TransformerMixin
import joblib, os
from datetime import datetime




class BCT_selection(TransformerMixin,BaseEstimator):
    """
        Boxplot Cutoff Thresholding (BCT).

        Key Idea : We augment a noise matrix to DataMatrix and fit the model on
        combined data. Then the upper threshold of outliers for noise variables can
        be considered as a cutoff for selection/rejection of original variables.

        Parameters
        ----------
        base_estimator : object having ``fit`` method, that provides attribute
        `feature_importances_` after fitting
            The base estimator used for feature selection.

        random_state : int ; default None
        Seed for resampling. It will only ensure reproducability of the augmented
        noise, and not of any randomness inside `base_estimator`.


        Attributes
        ----------
        BCT : float
            The cutoff in use. Any feature with importance exceeding this value
            will be selected, otherwise will be rejected.

        classes_ : array of shape (`n_classes_`,)
            A list of class labels known to the classifier.

        feature_importances_ : array of shape (`n_features_in_`,)
            Importances of features.

        feature_names_in_ : array of shape (`n_features_in_`,)
            Names of features seen during ``fit``.

        features_selected_ : array of shape (`n_features_selected_`,)
            Names of selected features.

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


        References
        ----------
        ..[1] Akarachantachote, Noppamas, Seree Chadcham, and Kidakan Saithanu.
        "Cutoff threshold of variable importance in projection for variable selection."
        Int J Pure Appl Math 94.3 (2014): 307-322.

    """

    def __init__(self,base_estimator,*,random_state=None):
        self.base_estimator = base_estimator
        self.random_state = random_state


    def _augment_noise(self,X):
        """
        Shuffles each column of X to generate noise matrix.

        [ for internal use only ]

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        DataFrame of shape (n_samples,2*n_features)
            [X,X_noise]

        """
        rng = np.random.default_rng(self.random_state)
        shuffle = lambda x : rng.permutation(x)
        X_noise = X.apply(shuffle,axis=0)
        X.columns = X.columns.astype(str)
        X_noise.columns = [name+'_noise' for name in X.columns]
        return pd.concat([X,X_noise],axis=1)


    def fit(self,X,y,**fit_params):
        """
        ``fit`` method for ``BCT_selection``.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples.

            NOTE : Each feature must be on same scale.

        y : Series of shape (n_samples,)
            The target values.

        **fit_params : other keyword arguments to the ``fit`` method of `base_estimator`.

        Returns
        -------
        self
            The fitted ``BCT_selection``.

        """
        ## adding noise matrix ....
        Z,y = self._augment_noise(pd.DataFrame(X)),pd.Series(y)
        ## fitting base estimator ....
        self.base_estimator.fit(Z,y,**fit_params)
        self.n_samples_,self.n_features_in_ = (self.base_estimator.n_samples_,
                                               self.base_estimator.n_features_in_//2)
        self.classes_,self.n_classes_ = (self.base_estimator.classes_,
                                         self.base_estimator.n_classes_)
        importances_ = self.base_estimator.feature_importances_.reshape((2,-1))
                        # 1st row is important, 2nd row is noise in `importances_`
        self.feature_importances_ = importances_[0]
        self.ranking_ = rankdata(-self.feature_importances_,method='ordinal')
        self.feature_names_in_ = self.base_estimator.feature_names_in_[:self.n_features_in_]
        ## BCT = _Q3 + 1.5(_Q3-_Q1) ....
        _Q1,_Q3 = np.quantile(importances_[1],0.25),np.quantile(importances_[1],0.75)
        self.BCT = 2.5*_Q3 - 1.5*_Q1
        ## remove noise from base estimator ....
        reset = {'n_features_in_':self.n_features_in_,
                 'feature_importances_':self.feature_importances_,
                 'threshold_':self.BCT,
                 'ranking_':self.ranking_,
                 'feature_names_in_':self.feature_names_in_}
        for key,value in reset.items() :
            setattr(self.base_estimator,key,value)
        return self


    def get_support(self,indices=False):
        """
        Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool ; default False
            If True, the return value will be an array of integers, rather than a boolean mask.

        Returns
        -------
        array
            An index that selects the retained features from a feature vector.
            If indices is False, this is a boolean array of shape [# input features],
            in which an element is True iff its corresponding feature is selected for retention.
            If indices is True, this is an integer array of shape [# output features] whose values
            are indices into the input feature vector.

        """
        out = self.base_estimator.get_support(indices)
        self.n_features_selected_ = self.base_estimator.n_features_selected_
        self.features_selected_ = getattr(self.base_estimator,'features_selected_',None)
        return out


    def transform(self,X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The input samples to be transformed.

        Returns
        -------
        DataFrame of shape (n_samples, n_selected_features)
            The input samples with only the selected features.

        """
        out = self.base_estimator.transform(X)
        self.n_features_selected_ = self.base_estimator.n_features_selected_
        self.features_selected_ = getattr(self.base_estimator,'features_selected_',None)
        return out


    def fit_transform(self,X,y,**fit_params):
        """
        ``fit`` the data (X,y) then ``transform`` X.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples.

        y : Series of shape (n_samples,)
            The target values.

        **fit_params : other keyword arguments to ``fit`` method.

        Returns
        -------
        DataFrame of shape (n_samples, n_selected_features)
            The input samples with only the selected features.

        """
        return self.fit(X,y,**fit_params).transform(X)


    def inverse_transform(self,X):
        """
        Reverse the transformation operation.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_selected_features)
            The input samples with only selected feature-columns.

        Returns
        -------
        DataFrame of shape (n_samples, `n_features_in_`)
            Columns of zeros inserted where features would have been removed by ``transform()``.

        """
        return self.base_estimator.inverse_transform(X)


    def plot(self,sort=True,savefig=None,**kwargs):
        """
        Make plot of `feature_importances_`.

        Colour Code :

            color[0](default 'green') is selected,
            color[1](default 'red') is rejected,
            (a stripe on colour implies false selection/rejection , when `true_support` is known).

        Parameters
        ----------
        sort : bool, default True
            Whether to sort the features according to `feature_importances_` before plotting.

        savefig : "directory/for/saving/your_plot"
            default None, implies plot will not be saved. True will save the plot inside a folder PLOTs at the current working directory.
            The plot will be saved as self.__class__.__name__-datetime.now().strftime('%Y_%m_%d_%H%M%S%f').png

        **kwargs : keyword arguments to pass to matplotlib plotting method.

        """
        kwargs.update({'ylabel':'BCT_'+ascii(self.base_estimator)})
        self.base_estimator.plot(sort=sort,savefig=savefig,**kwargs)


    def get_error_rates(self,true_imp,*,plot=False):
        """
        Computes various error-rates when true importance of the features are known.

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
        return self.base_estimator.get_error_rates(true_imp,plot=plot)


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
            The default is None, which will save the file at the current working directory
            as self-datetime.now().strftime('%Y_%m_%d_%H%M%S%f').pkl

        """
        if file_path is None :
            file_path = os.path.join(os.getcwd(),
                                     ascii(self)+f"-{datetime.now().strftime('%Y_%m_%d_%H%M%S%f')}.pkl")
        self.base_estimator.dump_to_file(file_path)




#### ==========================================================================
##
###
####
###
##
#### PIMP =====================================================================

class PIMP_selection(TransformerMixin,BaseEstimator):
    """
        References
        ----------
        ..[1] Altmann, André, et al. "Permutation importance: a corrected feature importance measure."
        Bioinformatics 26.10 (2010): 1340-1347.
    """


