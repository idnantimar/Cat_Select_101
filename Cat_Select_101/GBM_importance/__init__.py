"""
Created on Tue Jan  9 23:44:56 2024

Topic: The ``CatBoost``,being widely used in machine learning, provides some implementations of feature importances,
       which can be used as a baseline for comparison.

@author: R.Nandi
"""



#### CatBoost importance ======================================================

from catboost import CatBoostClassifier
from .. import My_Template_FeatureImportance




class cb_importance(My_Template_FeatureImportance):
    """
        Feature selection using ``CatBoost``.

        Class Variables
        ---------------
        Estimator_Type : ``CatBoostClassifier``

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

    Estimator_Type = CatBoostClassifier

    def __init__(self,random_state=None,*,
                 max_features=None,threshold=1e-10):
        super().__init__(random_state)
        self.threshold = threshold
        self.max_features = max_features
