"""
Created on Fri Dec 15 11:23:30 2023

Topic: LASSO based Variable Selection, using default configuration of sklearn.
       Since the wide popularity of LASSO , it is used as a baseline for comparison here.

@author: R.Nandi
"""



#### Vanilla LASSO ============================================================

from sklearn.linear_model import LogisticRegressionCV
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from . import My_Template_FeatureImportance



class vanillaLASSO_importance(My_Template_FeatureImportance):
    """
        Feature selection using Logistic LASSO regression with default configuration of ``LogisticRegressionCV``.

        Class Variables
        ---------------
        Estimator_Type : ``LogisticRegressionCV(penalty='l1',multi_class='multinomial',solver='saga',scoring=None,cv=None,Cs=10)``

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

    """

    Estimator_Type = LogisticRegressionCV(penalty='l1',multi_class='multinomial',
                                          solver='saga',
                                          scoring=None,cv=None,Cs=10)
    def __init__(self,random_state=None,*,
                 max_features=None,threshold=1e-10):
        super().__init__(random_state)
        self.threshold = threshold
        self.max_features = max_features


    def fit(self,X,y,*,n_jobs=None,max_iter=100,reduce_norm=1):
        """
        ``fit`` method for LASSO.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : Series of shape (n_samples,)
            The target values.

        n_jobs : int ; default None
            Number of CPU cores used during the cross-validation loop in ``LogisticRegressionCV``.

        max_iter : int ; default 100
            Maximum number of iterations of the optimization algorithm.

        reduce_norm : non-zero int, inf, -inf ; default 1
            Order of the norm used to compute `feature_importances_` in the case where the `coef_` of the
            ``LogisticRegressionCV`` is of dimension 2. By default 'l1'-norm is being used.

        Returns
        -------
        self
            The fitted ``vanillaLASSO_importance`` instance is returned.

        """
        self.estimator = clone(self.Estimator_Type)
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        super().fit(X,y)
        self.estimator.set_params(**{'n_jobs':n_jobs,'max_iter':max_iter,'random_state':self.random_state})
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        self.estimator.fit(X,y)
        warnings.filterwarnings("default", category=ConvergenceWarning)
        self.coef_ = self.estimator.coef_
        self.C_ = self.estimator.C_
        self.Cs_ = self.estimator.Cs_
        self.reduce_norm = reduce_norm
        self.feature_importances_ = self._coef_to_importance(self.reduce_norm,
                                                             identifiability=False)
        return self


    def transform(self,X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        2-D array of shape (n_samples, n_selected_features)
            The input samples with only the selected features.

        """
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        return super().transform(X)


    def get_error_rates(self,true_coef,*,plot=False):
        """
        Computes various error-rates when true importance of the features are known.

        Parameters
        ----------
        true_coef : array of shape (`n_features_in_`,) or (`n_classes_`,`n_features_in_`)
            If a 1-D boolean array , True implies the feature is important in true model, null feature otherwise.
            If a 1-D array of floats , it represent the `feature_importances_` of the true model,
            2-D array of floats represnt `coef_` of the true model.

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
            true_support = np.linalg.norm(self.true_coef.reshape(-1,self.n_features_in_),
                                          ord=self.reduce_norm,axis=0)
            self.true_support = (true_support > self.threshold_)
        return super().get_error_rates(plot=plot)





#### ==========================================================================
