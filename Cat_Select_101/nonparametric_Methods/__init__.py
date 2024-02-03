"""
Created on Tue Jan  9 23:44:56 2024

Topic: The ``CatBoost``,being widely used in machine learning,
       provides some implementations of feature importances,
       which can be used as a baseline for comparison.

@author: R.Nandi
"""



#### Gradient Boosting Machines importance ====================================

from catboost import CatBoostClassifier
from sklearn.base import clone
from .. import My_Template_FeatureImportance,_Data_driven_Thresholding
from .._utils.generate_Categorical import detect_Categorical



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

        iterations : int ; default 500
            Max count of trees.

        learning_rate : float in (0,1] ; default 0.03
            Step size shrinkage used in update to prevents overfitting.

        verbose : bool ; default False
            The verbosity level of training.

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

        configuration_cb : dict ; default {'depth':6,'l2_leaf_reg':3.0}
            dict of other keyword arguments to the initialization of ``CatBoostClassifier``.


        Attribures
        ----------
        cat_features_ : list
            Names of categorical features seen during ``fit``.

        classes_ : array of shape (`n_classes_`,)
            A list of class labels known to the classifier.

        confusion_matrix_for_features_ : array of shape (`n_features_in_`, `n_features_in_`)
            ``confusion_matrix`` (`true_support`, `support_`)

        estimator : a fitted ``CatBoostClassifier(...)`` instance

        false_discoveries_ : array of shape (`n_features_in_`,)
            Boolean mask of false positives.

        false_negatives_ : array of shape (`n_features_in_`,)
            Boolean mask of false negatives.

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

        support_ : array of shape (`n_features_in_`,)
            Boolean mask of selected features.

        threshold_ : float
            Cut-off in use, for selection/rejection.

        true_support : array of shape (`n_features_in_`,)
            Boolean mask of active features in population, only available after
            ``get_error_rates`` method is called with true_imp.


    """

    _permutation_importance = None
    _coef_to_importance = None
    Estimator_Type = CatBoostClassifier(loss_function='MultiClass',
                                        boosting_type='Ordered',grow_policy='SymmetricTree')
    def __init__(self,random_state=None,*,iterations=500,learning_rate=0.03,
                 verbose=False,
                 max_features=None,threshold=0,cumulative_score_cutoff=0.05,
                 configuration_cb={'depth':6,'l2_leaf_reg':3.0}):
        super().__init__(random_state)
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.threshold = threshold
        self.max_features = max_features
        self.cumulative_score_cutoff = cumulative_score_cutoff
        self.configuration_cb = configuration_cb


    def _return_cat(self,X):
        """
        method to detect the names of the categorical features.

        [ For internal use only. Override if necessary. ]

        """
        self.cat_features_ = list(detect_Categorical(X,get_names=True))


    def fit(self,X,y,**fit_params):
        """
        ``fit`` method for ``cb_importance``.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            they will be detected by ``_return_cat`` method.

        y : Series of shape (n_samples,)
            The target values.

        **fit_params : other keyword arguments to ``CatBoostClassifier.fit()``

        Returns
        -------
        self
            The fitted ``cb_importance`` instance is returned.

        """
        super().fit(X,y)
        self.configuration_cb.update({'iterations':self.iterations,'learning_rate':self.learning_rate,
                                 'verbose':self.verbose,
                                 'random_seed':self.random_state,
                                 'class_names':self.classes_})
        estimator = clone(self.Estimator_Type)
        estimator.set_params(**self.configuration_cb)
        self._return_cat(X)
        fit_params.update({'cat_features':self.cat_features_})
        estimator.fit(X,y,**fit_params)
        self.estimator = estimator
        self.feature_importances_ = self.estimator.feature_importances_
        _Data_driven_Thresholding(self)
        return self


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





