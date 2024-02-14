"""
Created on Sun Dec 24 14:01:32 2023

Topic: My Template of Feature Importance

@author: R.Nandi
"""



#### My_Template ==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator,check_is_fitted
from sklearn.feature_selection import SelectorMixin
from scipy.stats import rankdata
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import joblib, os
from datetime import datetime





class My_Template_FeatureImportance(SelectorMixin,BaseEstimator):
    """
        A common template for all the feature-importance techniques under this projectwork.

        Parameters
        ----------
        random_state : int , default None
            Seed for reproducible results across multiple function calls.


        Class Variables
        ---------------
        threshold : 0
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise will be rejected.

     """

    threshold = 0
    def __init__(self,random_state=None):
        self.random_state = random_state



    def fit(self,X,y):
        """
        This is a basic ``fit`` method , computing atributes `n_samples_`,`n_features_in_`, `feature_names_in_`,
        `classes_`, `n_classes_`.

        [ Override this with the actual implementation of required feature-importance technique ,
        that computes the attribute `feature_importances_` ]

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        """
        ## delete 'feature_importances_' from previous fit ...
        if hasattr(self,'feature_importances_') :
            delattr(self,'feature_importances_')
        ## measure dimensions ...
        self.n_samples_,self.n_features_in_ = getattr(X,'shape')
        self.feature_names_in_ = getattr(X,'columns',None)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)



    def _coef_to_importance(self,coef,reduce_norm=1,*,identifiability=False):
        """
        Computes `feature_importances_` based on `coef`.

        Parameters
        ----------
        coef : the fitted `coef_` or true coefficients of the model.

        reduce_norm : non-zero int, inf, -inf ; default 1
            Order of the norm used to compute `feature_importances_` in the case where the `coef`
            is of dimension 2. By default 'l1'-norm is being used.

        identifiability : bool ; default False
            Whether the `coef` are all identifiable or not. If `coef`
            is of shape (`n_classes_`,`n_featuress_in_`) set False, set True when
            there is already some baseline category, i.e. `coef` is of shape (`n_classes_-1`,`n_featuress_in_`).

        Returns
        -------
        1-D array of shape (`n_featuress_in_`,)

        """
        if not identifiability :
            # taking average over possible base-line categories ...
            k,m = coef.shape
            out = np.zeros((m,))
            for row in coef:
                out += np.linalg.norm(coef-row,ord=reduce_norm,axis=0)
            out /= k
        else :
            # base-line category already specified ...
            out = np.linalg.norm(coef,ord=reduce_norm,axis=0)
        ## feature importances ....
        return out



    def _permutation_importance(self,test_data,*,n_repeats=10,scoring=None,cumulative_score_cutoff=0.05):
        """
        This function calculates permutation based feature importances, assuming
        there is a fitted ``estimator``.

        Key Idea : Fit a model based on all features, then every time randomly permute observations of one feature column,
        keeping the other columns fixed, to break the association between that feature and response. Evaluate the
        performance of the fitted model once on permuted data and once on unpermuted data. The more
        important a feature is, the larger will be the corresponding drop in performance after permutation.

        [ Note: Calling this function will override the `coef_` based `feature_importances_` ]

        Parameters
        ----------
        test_data : tuple (X_test,y_test)
            X_test has shape (n_samples,n_features)
            y_test has shape (n_samples,)

        n_repeats : int ; default 10
            Number of times to permute a feature.

        scoring : str or callable ; default None
            Scorer to use.

        Returns
        -------
        A ``sklearn.inspection.permutation_importance`` object.

        """
        X_test,y_test = test_data
        out = permutation_importance(self.estimator,X_test,y_test,
                                     scoring=scoring,
                                     n_repeats=n_repeats,
                                     random_state=self.random_state)
        self.feature_importances_ = np.median(out.importances,axis=1)
        return out



    def _ranking(self):
        """
        The most important feature has rank=1,
        The least important feature has rank='n_features_in_'.

        If two features get exactly same importance (very rare),
        they will still get distinct integer ranks, based on which one occured first

        [ for internal use only ]


        """
        self.ranking_ = rankdata(-self.feature_importances_,method='ordinal')



    def _bound_max_features(self,max_features):
        """
        Constrain the maximum possible number of selection by a given constant

        Parameters
        ----------
        max_features : int ; default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

        [ 'threshold_' will be updated inplace ]

        """
        if (self.max_features is not None) and (self.max_features < self.n_features_in_) :
            self.threshold_ = max(self.feature_importances_[np.where(self.ranking_==(self.max_features+1))],
                                  self.threshold_)



    def _cumulative_score_threshold(self,cut_off=0.01):
        """
        'threshold_' for selecting those features that contributes to top
        100*(1-cut_off)% feature importances.

        Parameters
        ----------
        cut_off : float in [0,1) ; default 0.01

        [ 'threshold_' will be updated inplace ]

        NOTE : Result is not valid when all features
        are unimportant.

        """
        cum_importances_ = np.sort(self.feature_importances_).cumsum()
        n_select = (cum_importances_ >= cut_off*cum_importances_[-1]).sum()
        if n_select < self.n_features_in_ :
            self.threshold_ = max(self.feature_importances_[np.where(self.ranking_==(n_select+1))],
                                  self.threshold_)



    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected.

        Returns
        -------
        boolean array of shape (`n_features_in_`,)
            An element is True iff its corresponding feature is selected for
            retention.

        """
        check_is_fitted(self,attributes="feature_importances_",
                        msg="The %(name)s instance must have a 'feature_importances_' attribute.")
        self.support_ = (self.feature_importances_ > self.threshold_)
        self.n_features_selected_ = self.support_.sum()
        if hasattr(self,'feature_names_in_') :
            self.features_selected_ = self.feature_names_in_[self.support_]
        return self.support_



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
        X_tr = super().transform(X)
        return pd.DataFrame(X_tr,
                            index=X.index,columns=getattr(self,'features_selected_',None))



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
        return super().fit_transform(X,y,**fit_params)



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
        X_r = super().inverse_transform(X)
        return pd.DataFrame(X_r,
                            index=X.index,columns=getattr(self,'feature_names_in_',None))



    def plot(self,sort=True,savefig=None,*,kind='bar',ax=None,
             xlabel='features',ylabel=None,title=None,rot=30,color=['green','red'],**kwargs):
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

        kind, ax, xlabel, ylabel, title, rot, color, **kwargs : keyword arguments to pass to matplotlib plotting method.

        """
        self.get_support()
        if self.feature_names_in_ is None :
            ix = (np.vectorize(lambda j:"X_"+str(j)))(range(self.n_features_in_))
        else : ix = self.feature_names_in_
        if ylabel is None :
            ylabel = self.__class__.__name__
        if title is None :
            title = "selected : " + str(self.n_features_selected_) +"/" + str(self.n_features_in_)
        imp = pd.Series(self.feature_importances_,index=ix)
        colors = np.array([(color[0] if val else color[1]) for val in self.support_])
        truth_known = hasattr(self,'true_support')
        if truth_known :
            hatch_patterns = np.array([('/' if val else None) for val in (self.true_support!=self.support_)])
        else : hatch_patterns = np.array([None]*self.n_features_in_)
        if sort :
            sort_ix = np.argsort(-self.feature_importances_)
            imp = imp.iloc[sort_ix]
            colors = colors[sort_ix]
            hatch_patterns = hatch_patterns[sort_ix]
        imp.plot(kind=kind,ax=ax,xlabel=xlabel,ylabel=ylabel,title=title,rot=rot,
                 color=colors,hatch=hatch_patterns,**kwargs)
                ## in default plots, red: rejected, green: selected , stripe: false +-
        plt.axhline(self.threshold_,color='black',linestyle='dashed')
        if savefig is not None :
            if savefig==True :
                os.makedirs('PLOTs',exist_ok=True)
                savefig = 'PLOTs'
            plt.savefig(os.path.join(savefig,
                                     f"{self.__class__.__name__}-{datetime.now().strftime('%Y_%m_%d_%H%M%S%f')}.png"))
        plt.show()



    def _get_error_rates(self,*,plot=False):
        """
        This function computes `pfer_`, `pcer_`, `fdr_`, `false_discoveries_`,
        `minimum_model_size_`, `tpr_`, `n_false_negatives_`, `confusion_matrix_for_features_`,
        `f1_score_for_features_`, `youden_index_for_features_` assuming there is an attribute `true_support`.

        Can plot the `confusion_matrix_for_features_`.

        [ Override this with the actual implementation of computing `true_support`,
        based on the true model coefficients input, if known ]

        Returns
        -------
        dict
            Conatins various error rates PCER,FDR,PFER,TPR etc.

        """
        _compare_truth = (np.array(self.true_support,dtype=int)-self.support_)
        self.false_discoveries_ = (_compare_truth == -1)
                ## if a feature is True in 'support_' and False in 'true_support'
                 ## it is a false-discovery or false +ve
        minimum_model_size_ = (self.ranking_[self.true_support]).max()
                ## from the ordering of 'feature_importances_' the minimum number of features to be selected
                 ## to include the least true important features and those false discoveries having more importance
                  ## same as computing the maximum 'ranking_' among true important features
        minimum_model_size_ /= np.sum(self.true_support)
                ## instead of reporting the raw number , the ratio to actual size is reported
                ## this ratio always >=1 , 1 is the ideal case
        pfer_ = self.false_discoveries_.sum()
        pcer_ = 100*pfer_/self.n_features_in_
        fdr_ = 100*(1 - precision_score(y_true=self.true_support,y_pred=self.support_,
                                         zero_division=1.0))
                ## lower 'fdr_' is favourable
        self.false_negatives_ = (_compare_truth == 1)
                ## if a feature is False in 'support_' and True in 'true_support'
                 ## it is a false -ve
        n_false_negatives_ = self.false_negatives_.sum()
        tpr_ = recall_score(y_true=self.true_support,y_pred=self.support_,
                                 zero_division=np.nan)
                ## higher 'tpr_' is favourable
        self.confusion_matrix_for_features_ = confusion_matrix(y_true=self.true_support,
                                                               y_pred=self.support_)
        f1_score_for_features_ = f1_score(y_true=self.true_support,y_pred=self.support_,
                                               zero_division=np.nan)
        _n_negatives = (self.n_features_in_-np.sum(self.true_support))
        youden_index_for_features_ = (tpr_ - (pfer_/_n_negatives)) if _n_negatives else None
                ## this confusion matrix or f1 score or youden index corresponds to the labelling of
                 ## null/non-null features, not corresponds to the labelling of target(y) classes
        tpr_ *= 100
        if plot :
            ConfusionMatrixDisplay(self.confusion_matrix_for_features_,
                                   display_labels=['null','non-null']).plot(colorbar=False)
        return {'PCER':pcer_,
                'FDR':fdr_,
                'PFER':pfer_,
                'TPR':tpr_,
                'n_FalseNegatives':n_false_negatives_,
                'minModel_size_ratio':minimum_model_size_,
                'selection_F1':f1_score_for_features_,
                'selection_YoudenJ':youden_index_for_features_}



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
            The default is None, which will save the file at the current working directory as self.__class__.__name__-datetime.now().strftime('%Y_%m_%d_%H%M%S%f').pkl

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
##
###
####
###
##
#### Data based Threshold =====================================================


def _Data_driven_Thresholding(self,Steps=[0,1,2,3]):
    """
        Step0:  Resets `threshold_` to default value.
        Step1:  Ranks the features based on their importances.
        Step2:  Computes `threshold_` based on cumulative scores.
        Step3:  Updates `threshold_` based on maximum number of selection (if any).

        [ to avoid rewriting common steps again and again ]

        NOTE: Based on same data (X,y) , if we want to compute `threshold_` for
        different `max_features` or `cumulative_score_cutoff` we do not need to refit
        the model since `feature_importances_` are not subject to change.

        Example -
            >>> model = vanilla_LASSO_importance(max_features=None)
            >>> model.fit(X,y)  ## computing feature importances
            >>> model.get_support()  ## feature selection for the first time
            >>> setattr(model,'max_features',5)
            >>> _Data_driven_Thresholding(model)
            >>> model.get_support()  ## feature selection for the updated setup

    """
    ToDo = {0:lambda : setattr(self,'threshold_',self.threshold),
            1:lambda : self._ranking(),
            2:lambda : self._cumulative_score_threshold(self.cumulative_score_cutoff),
            3:lambda : self._bound_max_features(self.max_features)}
    for step in Steps : ToDo[step]()




#### ==========================================================================
