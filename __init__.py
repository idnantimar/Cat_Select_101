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

        max_features : None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

    """
    threshold = 0
    max_features = None
    def __init__(self,random_state=None):
        self.random_state = random_state



    def fit(self,X,y,*,y_classes=True):
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

        y_classes : bool , default True
            Whether to obtain `classes_` and `n_classes_` for classification problems.

        """
        if hasattr(self,'feature_importances_') :
            delattr(self,'feature_importances_')
                ## delete 'feature_importances_' from previous fit
        self.n_samples_,self.n_features_in_ = getattr(X,'shape')
        self.feature_names_in_ = getattr(X,'columns',None)
        if y_classes :
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)



    def _coef_to_importance(self,reduce_norm=1,*,identifiability=False):
        """
        Computes `feature_importances_` based on `coef_`.

        Parameters
        ----------
        reduce_norm : non-zero int, inf, -inf ; default 1
            Order of the norm used to compute `feature_importances_` in the case where the `coef_`
            is of dimension 2. By default 'l1'-norm is being used.

        identifiability : bool ; default False
            Whether the estimated ``coef_`` are all identifiable or not. If ``coef_``
            is of shape (``n_classes_``,``n_featuress_in_``) set False, set True when
            ``coef_`` is of shape (``n_classes_-1``,``n_featuress_in_``).

        Returns
        -------
        1-D array of shape (``n_featuress_in_``,)

        """
        coef = self.coef_
        if not identifiability :
            coef = coef[:-1] - coef[-1]
        return np.linalg.norm(coef,ord=reduce_norm,axis=0)



    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected.

        Returns
        -------
        support : boolean array of shape (``n_features_in_``,)
            An element is True iff its corresponding feature is selected for
            retention.

        """
        check_is_fitted(self,attributes="feature_importances_",
                        msg="The %(name)s instance must have a 'feature_importances_' attribute.")

        ranking_ = rankdata(-self.feature_importances_,method='ordinal')
                ## the most important feature has rank=1
                 ## the least important feature has rank='n_features_in_'
                  ## if two features get exactly same importance (very rare),
                  ## they will still get distinct integer ranks, based on which one occured first
        if (self.max_features is None) :
            self.threshold_ = self.threshold
        else:
            cut_off = np.where(ranking_==self.max_features)
            self.threshold_ = max(self.feature_importances_[cut_off],
                                  self.threshold)
                ## constrain the maximum possible number of selection
                 ## by a given constant
        self.support_ = (self.feature_importances_ >= self.threshold_)
        self.ranking_ = ranking_
        self.n_features_selected_ = self.support_.sum()
        return self.support_



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
        if not hasattr(self,'threshold_') : self.get_support()
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



    def get_error_rates(self,*,plot=False):
        """
        This function computes attributes `pfer_`, `pcer_`, `fdr_`, `false_discoveries_`,
        `minimum_model_size_`, `tpr_`, `n_false_negatives_`, `confusion_matrix_for_features_`,
        `f1_score_for_features_` assuming there is an attribute `true_support`.

        Can plot the `confusion_matrix_for_features_`.

        [ Override this with the actual implementation of computing `true_support`,
        based on the true model coefficients input, if known ]

        Returns
        -------
        dict
            Conatins various error rates PCER,FDR,PFER,TPR.

        """
        compare_truth = (np.array(self.true_support,dtype=int)-self.support_)
        self.false_discoveries_ = (compare_truth == -1)
                ## if a feature is True in 'support_' and False in 'true_support'
                 ## it is a false-discovery or false +ve
        self.minumum_model_size_ = (self.ranking_[self.true_support]).max()
                ## from the ordering of 'feature_importances_' the minimum number of features to be selected
                 ## to include the least true important features and those false discoveries having more importance
                  ## same as computing the maximum 'ranking_' among true important features
        self.pfer_ = self.false_discoveries_.sum()
        self.pcer_ = self.pfer_/self.n_features_in_
        self.fdr_ = 1 - precision_score(y_true=self.true_support,y_pred=self.support_,
                                         zero_division=1.0)
                ## lower 'fdr_' is favourable
        self.false_negatives_ = (compare_truth == 1)
                ## if a feature is False in 'support_' and True in 'true_support'
                 ## it is a false -ve
        self.n_false_negatives_ = self.false_negatives_.sum()
        self.tpr_ = recall_score(y_true=self.true_support,y_pred=self.support_,
                                 zero_division=np.nan)
                ## higher 'tpr_' is favourable
        self.confusion_matrix_for_features_ = confusion_matrix(y_true=self.true_support,
                                                               y_pred=self.support_)
        self.f1_score_for_features_ = f1_score(y_true=self.true_support,y_pred=self.support_,
                                               zero_division=np.nan)
                ## this confusion matrix or f1 score corresponds to the labelling of
                 ## null/non-null features, not corresponds to the labelling of target(y) classes
        if plot :
            ConfusionMatrixDisplay(self.confusion_matrix_for_features_,
                                   display_labels=['null','non-null']).plot(colorbar=False)
        return {'PCER':self.pcer_,
                'FDR':self.fdr_,
                'PFER':self.pfer_,
                'TPR':self.tpr_}



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

