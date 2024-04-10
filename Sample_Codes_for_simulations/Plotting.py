"""
Created on Mon Feb 19 19:31:37 2024

Topic: Effect of Sample Sizes
@author: R.Nandi
"""

# working dir & useful Modules ..........................
import os
os.chdir("E:\SIMULATIONs\Simulation004_5Apr\COMPARISONs\Comparison_01")
from sys import path
path.append("E:\\")


import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Cat_Select_101.LASSO_based_Methods import vanillaLASSO_importance,L1SVM_importance
from Cat_Select_101.nonparametric_Methods import Independence_Screening
from Cat_Select_101.other_PenalizedRegressions import discriminative_LeastSquares
METHODs = {'lasso':vanillaLASSO_importance(multi_class='multinomial',Cs=[1e-2,1e-1,1.,1e+1,1e+2],random_state=101),
           'l1svm':L1SVM_importance(Cs=[1e-2,1e-1,1.,1e+1,1e+2],cv_config={'cv':None,'verbose':0},random_state=101),
           'dls':discriminative_LeastSquares.dLS_impotance(regularization=[1e-2,1e-1,1.,1e+1,1e+2],cv_config={'cv':None,'verbose':0})}
METHODs_nopredict = {'sis':Independence_Screening.SIS_importance()}


m = 100
a = 10
sample_sizes = [128,256,512,1024]


BETAs = np.array(
      [[  9.051,   1.928,   4.537],
       [ -4.347,  -6.068, -11.687],
       [ 13.982,  -3.242,   8.17 ],
       [-10.17 ,  -5.041,   0.864],
       [  5.039,  11.612,  -4.136],
       [  4.314,   7.765,  -3.321],
       [ 10.748,  -9.567, -13.383],
       [ -0.836,  12.902,   3.738],
       [ -1.561,  -6.155,   1.74 ],
       [  7.465,  -5.714, -10.949]]
      ).T

def Plotting0(B):
    u0 = np.abs(B[[1,2]]-B[0])
    u1 = np.abs(B[[0,2]]-B[1])
    u2 = np.abs(B[[0,1]]-B[2])

    u = np.sum((u0+u1+u2)/3,axis=0)
    u = pd.Series(100*u/sum(u))
    u.iloc[:a].plot(kind='bar',title='proposed importance')





IMPORTANCEs = {}
with open(r"E:\SIMULATIONs\Simulation004_5Apr\CASEs\case-100-10-0128\IMPORTANCEs-2024_04_09_004109662254.pkl",'rb') as file :
    IMPORTANCEs['n_128'] = joblib.load(file)
with open(r"E:\SIMULATIONs\Simulation004_5Apr\CASEs\case-100-10-0256\IMPORTANCEs-2024_04_09_001201302336.pkl",'rb') as file :
    IMPORTANCEs['n_256'] = joblib.load(file)
with open(r"E:\SIMULATIONs\Simulation004_5Apr\CASEs\case-100-10-0512\IMPORTANCEs-2024_04_09_011510485213.pkl",'rb') as file :
    IMPORTANCEs['n_512'] = joblib.load(file)
with open(r"E:\SIMULATIONs\Simulation004_5Apr\CASEs\case-100-10-1024\IMPORTANCEs-2024_04_09_022540679924.pkl",'rb') as file :
    IMPORTANCEs['n_1024'] = joblib.load(file)


def Plotting1(METHODs):
    for size in sample_sizes:
        Imp = []
        for key,value in METHODs.items():
            Imp_ = IMPORTANCEs[f'n_{size}'][key]
            Imp_ = 100*Imp_.div(np.sum(Imp_,axis=1),axis=0)
            Imp_.columns = ['f_'+str(j) for j in range(m)]
            Imp_['max_null'] = np.max(Imp_[['f_'+str(j) for j in range(a,m)]],axis=1)
            Imp_ = Imp_.drop(columns=['f_'+str(j) for j in range(a,m)])
            Imp_['method'] = key
            Imp += [Imp_]
        Imp = pd.concat(Imp)
        axes = Imp.boxplot(by='method',layout=(1,a+1),figsize = (20,5),grid=False,rot=30)
        plt.suptitle(f'feature importances | #samples:{size}')
        for t,truth in enumerate(np.array([True]*a+[False],dtype=str)): axes[t].set_xlabel(truth)
        plt.show()


CHECKs = {}
with open(r"E:\SIMULATIONs\Simulation004_5Apr\CASEs\case-100-10-0128\CHECKs_PIMP-2024_04_09_004109677883.pkl",'rb') as file :
    CHECKs['n_128'] = joblib.load(file)
with open(r"E:\SIMULATIONs\Simulation004_5Apr\CASEs\case-100-10-0256\CHECKs_PIMP-2024_04_09_001201317957.pkl",'rb') as file :
    CHECKs['n_256'] = joblib.load(file)
with open(r"E:\SIMULATIONs\Simulation004_5Apr\CASEs\case-100-10-0512\CHECKs_PIMP-2024_04_09_011510510805.pkl",'rb') as file :
    CHECKs['n_512'] = joblib.load(file)
with open(r"E:\SIMULATIONs\Simulation004_5Apr\CASEs\case-100-10-1024\CHECKs_PIMP-2024_04_09_022540695298.pkl",'rb') as file :
    CHECKs['n_1024'] = joblib.load(file)


def Plotting2(METHODs):
    Check_Values = {'PCER':(pd.DataFrame(columns=METHODs.keys(),index=sample_sizes),
                            pd.DataFrame(columns=METHODs.keys(),index=sample_sizes)),
                    'PFER':(pd.DataFrame(columns=METHODs.keys(),index=sample_sizes),
                            pd.DataFrame(columns=METHODs.keys(),index=sample_sizes)),
                    'FDR':(pd.DataFrame(columns=METHODs.keys(),index=sample_sizes),
                           pd.DataFrame(columns=METHODs.keys(),index=sample_sizes)),
                    'TPR':(pd.DataFrame(columns=METHODs.keys(),index=sample_sizes),
                           pd.DataFrame(columns=METHODs.keys(),index=sample_sizes)),
                    'n_FalseNegatives':(pd.DataFrame(columns=METHODs.keys(),index=sample_sizes),
                                        pd.DataFrame(columns=METHODs.keys(),index=sample_sizes)),
                    'minModel_size_ratio':(pd.DataFrame(columns=METHODs.keys(),index=sample_sizes),
                                           pd.DataFrame(columns=METHODs.keys(),index=sample_sizes)),
                    'selection_F1':(pd.DataFrame(columns=METHODs.keys(),index=sample_sizes),
                                    pd.DataFrame(columns=METHODs.keys(),index=sample_sizes)),
                    'selection_YoudenJ':(pd.DataFrame(columns=METHODs.keys(),index=sample_sizes),
                                         pd.DataFrame(columns=METHODs.keys(),index=sample_sizes)),
                    'selection_proportion':(pd.DataFrame(columns=METHODs.keys(),index=sample_sizes),
                                            pd.DataFrame(columns=METHODs.keys(),index=sample_sizes))}

    for key,value in Check_Values.items():
        x_jitter = np.array([np.random.permutation(range(-len(METHODs),len(METHODs),2)) for size in sample_sizes])
        x_val = np.arange(len(sample_sizes))
        for idx,method in enumerate(METHODs):
            for size in sample_sizes:
                obs = CHECKs[f'n_{size}'][method][key]
                value[0].loc[size,method],value[1].loc[size,method] = np.median(obs),np.percentile(obs,75)-np.percentile(obs,25)
            plt.errorbar(x=x_val + 0.025*x_jitter[:,idx],
                         y=value[0][method],
                         yerr=value[1][method]/2,
                         label=method, fmt='-o',capsize=2.5)
        plt.title(f"Average {key}")
        plt.xlabel("n_samples")
        plt.ylabel(key)
        plt.xticks(x_val,sample_sizes)
        plt.legend()
        plt.grid(linestyle='--',axis='y')
        plt.show()



SCOREs = {}
with open(r"E:\SIMULATIONs\Simulation004_5Apr\CASEs\case-100-10-0128\SCOREs_PIMP-2024_04_09_004109677883.pkl",'rb') as file :
    SCOREs['n_128'] = joblib.load(file)
with open(r"E:\SIMULATIONs\Simulation004_5Apr\CASEs\case-100-10-0256\SCOREs_PIMP-2024_04_09_001201333616.pkl",'rb') as file :
    SCOREs['n_256'] = joblib.load(file)
with open(r"E:\SIMULATIONs\Simulation004_5Apr\CASEs\case-100-10-0512\SCOREs_PIMP-2024_04_09_011510514799.pkl",'rb') as file :
    SCOREs['n_512'] = joblib.load(file)
with open(r"E:\SIMULATIONs\Simulation004_5Apr\CASEs\case-100-10-1024\SCOREs_PIMP-2024_04_09_022540695298.pkl",'rb') as file :
    SCOREs['n_1024'] = joblib.load(file)





def Plotting3(METHODs):
    for key in METHODs:
        plt.figure(figsize=(8,4))
        shared_ax = None
        for idx,size in enumerate(sample_sizes):
            ax = plt.subplot(1, len(sample_sizes), 1 + idx, sharey=shared_ax)
            Scores = SCOREs[f'n_{size}'][key][['logi_score_full','logi_score_transform']]
            Scores.columns = ['full','transform']
            sns.boxplot(Scores,ax=ax)
            plt.title('#samples:'+str(size))
            plt.xticks(rotation=40)
            if shared_ax is None:
                shared_ax = ax
        plt.suptitle(key+' | logistic accuracy score')
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(8,4))
        shared_ax = None
        for idx,size in enumerate(sample_sizes):
            ax = plt.subplot(1, len(sample_sizes), 1 + idx, sharey=shared_ax)
            Scores = SCOREs[f'n_{size}'][key][['knn_score_full','knn_score_transform']]
            Scores.columns = ['full','transform']
            sns.boxplot(Scores,ax=ax)
            plt.title('#samples:'+str(size))
            plt.xticks(rotation=40)
            if shared_ax is None:
                shared_ax = ax
        plt.suptitle(key+' | knn accuracy score')
        plt.tight_layout()
        plt.show()

