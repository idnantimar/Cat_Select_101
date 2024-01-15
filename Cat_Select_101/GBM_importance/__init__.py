"""
Created on Tue Jan  9 23:44:56 2024

Topic: The ``CatBoost``,being widely used in machine learning, provides some implementations of feature importances,
       which can be used as a baseline for comparison.

@author: R.Nandi
"""



#### CatBoost importance ======================================================

from catboost import CatBoostClassifier
from sklearn.base import clone
from .. import My_Template_FeatureImportance



class cb_importance(My_Template_FeatureImportance):
    """
        Feature selection using ``CatBoost``.

        Class Variables
        ---------------
        Estimator_Type : ``CatBoostClassifier(loss_function='MultiClass',
                                              boosting_type='Ordered',grow_policy='SymmetricTree')``

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

    _permutation_importance = None
    _coef_to_importance = None
    Estimator_Type = CatBoostClassifier(loss_function='MultiClass',
                                        boosting_type='Ordered',grow_policy='SymmetricTree')
    def __init__(self,random_state=None,*,
                 max_features=None,threshold=1e-10):
        super().__init__(random_state)
        self.threshold = threshold
        self.max_features = max_features


    def fit(self,X,y,cat_features,*,
            iterations=500,learning_rate=0.03,
            verbose=False,
            configuration_cb={'depth':6,'l2_leaf_reg':3.0},
            **fit_params):
        """
        ``fit`` method for ``cb_importance``

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : Series of shape (n_samples,)
            The target values.

        cat_features : list or array
            The list of categorical columns indices.

        iterations : int ; default 500
            Max count of trees.

        learning_rate : float in (0,1] ; default 0.03
            Step size shrinkage used in update to prevents overfitting.

        verbose : bool ; default False
            The verbosity level of training.

        configuration_cb : dict ; default {'depth':6,'l2_leaf_reg':3.0}
            Other keyword arguments to the initialization of ``CatBoostClassifier``.

        **fit_params : other keyword arguments to ``CatBoostClassifier.fit()``

        Returns
        -------
        self
            The fitted ``cb_importance`` instance is returned.

        """
        super().fit(X,y)
        configuration_cb.update({'iterations':iterations,'learning_rate':learning_rate,
                                 'verbose':verbose,
                                 'random_seed':self.random_state,
                                 'class_names':self.classes_})
        estimator = clone(self.Estimator_Type)
        estimator.set_params(**configuration_cb)
        fit_params.update({'cat_features':list(cat_features)})
        estimator.fit(X,y,**fit_params)
        self.estimator = estimator
        self.feature_importances_ = self.estimator.feature_importances_
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
            self.true_support = (self.true_coef >= self.threshold_)
        return super().get_error_rates(plot=plot)




#### ==========================================================================

