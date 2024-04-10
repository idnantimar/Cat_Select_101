"""
Created on Thu Feb  8 15:15:04 2024

Topic: Simulation

@author: R.Nandi
"""

m,a,n = (60,
         10,
         256)
n_itr = 24
n_resampling = 100

# working dir & useful Modules ..........................
import os
os.chdir(r"E:\SIMULATIONs\Simulation004_5Apr\CASEs\case-060-10-0256")


from sys import path
path.append("E:\\")

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from datetime import datetime



from joblib import cpu_count
Parallel_itr = lambda base_fun,kwargs={'n_jobs':cpu_count(True)} : Parallel(**kwargs)(delayed(base_fun)(t) for t in range(n_itr))
RNG = lambda seed=None : np.random.default_rng(seed)
def dump_to_file(obj,name):
    file_path = os.path.join(os.getcwd(),
                                     name+f"-{datetime.now().strftime('%Y_%m_%d_%H%M%S%f')}.pkl")
    with open(file_path, 'wb') as file:
            joblib.dump(obj, file)
    print(f"Instance saved as {file_path} successfully...")



## Collecting Data =================================================


with open(r"E:\SIMULATIONs\Simulation004_5Apr\DATA-2024_04_08_210606435114.pkl", 'rb') as file:
    DATA = joblib.load(file)

with open(r"E:\SIMULATIONs\Simulation004_5Apr\test_DATA-2024_04_08_210611896262.pkl", 'rb') as file:
    test_DATA = joblib.load(file)

Truth = [True]*a + [False]*(m-a)

## ===============================================================



## Feature Importances ===========================================
from Cat_Select_101.LASSO_based_Methods import vanillaLASSO_importance,L1SVM_importance
from Cat_Select_101.nonparametric_Methods import Independence_Screening
from Cat_Select_101.other_PenalizedRegressions import discriminative_LeastSquares


METHODs = {'lasso':vanillaLASSO_importance(multi_class='multinomial',Cs=[1e-2,1e-1,1.,1e+1,1e+2],random_state=101),
           'l1svm':L1SVM_importance(Cs=[1e-2,1e-1,1.,1e+1,1e+2],cv_config={'cv':None,'verbose':0},random_state=101),
           'dls':discriminative_LeastSquares.dLS_impotance(regularization=[1e-2,1e-1,1.,1e+1,1e+2],cv_config={'cv':None,'verbose':0})}
METHODs_nopredict = {'sis':Independence_Screening.SIS_importance()}

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from Cat_Select_101._noise_based_Thresholding import PIMP_selection

_Update_Fit = {'lasso':lambda selector : selector.set_params(Cs=[selector.C_[0]]),
              'l1svm':lambda selector : selector.set_params(Cs=[selector.C_]),
              'dls':lambda selector : selector.set_params(regularization=[selector.best_penalty_]),
              'sis': lambda selector : selector}


def COMPUTE(METHODs):
    IMPORTANCEs = {}
    CHECKs = {}
    SCOREs = {}
    CHECKs_PIMP = {}
    SCOREs_PIMP = {}
    Pvals = {}

    for key,value in METHODs.items():
        def one_copy(itr):
            X,y = DATA[itr][f'm_{m}'][f'a_{a}']
            X,y = X.iloc[:n],y.iloc[:n]
            X_test,y_test = test_DATA[itr][f'm_{m}'][f'a_{a}']
            ## Base Method ....
            Selector = value
            Selector.fit(X,y)
            Estimators = {'logi_':LogisticRegression(penalty=None),
                          'knn_':KNeighborsClassifier(n_neighbors=int(np.sqrt(n)))}
            scores_ = {}
            for _key,_value in Estimators.items():
                _value.fit(X,y)
                scores_[_key+'score_full'] = _value.score(X_test,y_test)
            out = (Selector.feature_importances_,
                   {**Selector.get_error_rates(Truth),
                    **{'selection_proportion':100*Selector.n_features_selected_/Selector.n_features_in_}})
            if any(Selector.get_support()):
                X_tran = Selector.transform(X)
                testX_tran = Selector.transform(X_test)
                for _key,_value in Estimators.items():
                    _value.fit(X_tran,y)
                    scores_[_key+'score_transform'] = _value.score(testX_tran,y_test)
            else:
                for _key,_value in Estimators.items():
                    scores_[_key+'score_transform'] = 0.
            out += (scores_,)
            ## Permutation Test ....
            Selector_PIMP = PIMP_selection(_Update_Fit[key](Selector),
                                           n_resampling,
                                      multipletests='fdr_by')
            Selector_PIMP.fit(X,y)
            Estimators = {'logi_':LogisticRegression(penalty=None),
                          'knn_':KNeighborsClassifier(n_neighbors=int(np.sqrt(n)))}
            scores_PIMP = {}
            for _key,_value in Estimators.items():
                _value.fit(X,y)
                scores_PIMP[_key+'score_full'] = _value.score(X_test,y_test)

            out += (Selector_PIMP.p_value,
                   {**Selector_PIMP.get_error_rates(Truth),
                    **{'selection_proportion':100*Selector_PIMP.base_estimator.n_features_selected_/Selector_PIMP.n_features_in_}})
            if any(Selector.get_support()):
                X_tran = Selector_PIMP.transform(X)
                testX_tran = Selector_PIMP.transform(X_test)
                for _key,_value in Estimators.items():
                    _value.fit(X_tran,y)
                    scores_PIMP[_key+'score_transform'] = _value.score(testX_tran,y_test)
            else:
                for _key,_value in Estimators.items():
                    scores_PIMP[_key+'score_transform'] = 0.
            out += (scores_PIMP,)
            ## ....
            return out
        Out = Parallel_itr(one_copy,{'n_jobs':-1})

        Imp = []
        Check = []
        Score = []
        for itr in range(n_itr):
            Imp += [Out[itr][0]]
            Check += [Out[itr][1]]
            Score += [Out[itr][2]]
        IMPORTANCEs[key] = pd.DataFrame(Imp)
        CHECKs[key] = pd.DataFrame(Check)
        SCOREs[key] = pd.DataFrame(Score)

        p_val = []
        Check = []
        Score = []
        for itr in range(n_itr):
            p_val += [Out[itr][3]]
            Check += [Out[itr][4]]
            Score += [Out[itr][5]]
        Pvals[key] = pd.DataFrame(p_val)
        CHECKs_PIMP[key] = pd.DataFrame(Check)
        SCOREs_PIMP[key] = pd.DataFrame(Score)


    return IMPORTANCEs,CHECKs,SCOREs,Pvals,CHECKs_PIMP,SCOREs_PIMP





IMPORTANCEs,CHECKs,SCOREs,p_values_PIMP,CHECKs_PIMP,SCOREs_PIMP = COMPUTE({**METHODs,**METHODs_nopredict})

dump_to_file(IMPORTANCEs,'IMPORTANCEs')
dump_to_file(CHECKs,'CHECKs')
dump_to_file(SCOREs,'SCOREs')
dump_to_file(p_values_PIMP,'p_values_PIMP')
dump_to_file(CHECKs_PIMP,'CHECKs_PIMP')
dump_to_file(SCOREs_PIMP,'SCOREs_PIMP')



## =============================================================

