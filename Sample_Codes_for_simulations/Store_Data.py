"""
Created on Mon Mar 18 16:02:22 2024

Topic: Simulation

Case:

            Features         |  Observations  |  Target Class
    -----------------------------------------------------------
        i) Independent       |      iid       |      3
        ii) Numerical        |                |
        iii) N(0,1)          |                |
                             |                |
    -----------------------------------------------------------

@author: R.Nandi
"""

# working dir & useful Modules ..........................
import os
os.chdir("E:\SIMULATIONs\Simulation004_5Apr")
from sys import path
path.append("E:\\")

import numpy as np
import joblib
from joblib import Parallel, delayed
from datetime import datetime



## Simulating Data =================================================
sample_size = 1024 # [128,256,512,1024]
n_features = [30,60,100,150] # in increasing order
n_active = {30:[1,5,10],
            60:[1,5,10,25],
            100:[1,5,10,25,40],
            150:[1,5,10,25,40]} # in increasing order
class_labels = ['cat1','cat2','cat3']
n_itr = 24

from joblib import cpu_count
Parallel_itr = lambda base_fun,kwargs={'n_jobs':cpu_count(True)} : Parallel(**kwargs)(delayed(base_fun)(t) for t in range(n_itr))
RNG = lambda seed=None : np.random.default_rng(seed)
def dump_to_file(obj,name):
    file_path = os.path.join(os.getcwd(),
                                     name+f"-{datetime.now().strftime('%Y_%m_%d_%H%M%S%f')}.pkl")
    with open(file_path, 'wb') as file:
            joblib.dump(obj, file)
    print(f"Instance saved as {file_path} successfully...")



BETA_coeffs = np.array(
      [[  9.051,   1.928,   4.537],
       [ -4.347,  -6.068, -11.687],
       [ 13.982,  -3.242,   8.17 ],
       [-10.17 ,  -5.041,   0.864],
       [  5.039,  11.612,  -4.136],
       [  4.314,   7.765,  -3.321],
       [ 10.748,  -9.567, -13.383],
       [ -0.836,  12.902,   3.738],
       [ -1.561,  -6.155,   1.74 ],
       [  7.465,  -5.714, -10.949],
       [-11.934,  -1.648,  10.999],
       [ -7.844,  14.959,  -7.493],
       [ 14.092,  11.883,  14.08 ],
       [  5.534,   5.582,  -9.507],
       [  1.033,  13.631,  -3.748],
       [-12.756,  13.798, -12.166],
       [  9.102, -14.711,  12.777],
       [ -1.592,   3.423,  -2.299],
       [-12.962,  -2.059,   9.063],
       [ -1.141,  -2.807,  13.53 ],
       [ -2.211, -11.114,  -1.058],
       [  1.453,  -1.057,  -3.787],
       [ 13.571,  13.186,   0.078],
       [ -7.841,  -8.436,  -3.474],
       [ 12.524,  -5.994,  -4.767],
       [-10.529,   6.867,  12.191],
       [ -4.539,   3.019,   3.581],
       [ -9.623, -14.051, -14.84 ],
       [ -6.025,   7.75 ,   8.433],
       [ -7.715, -11.205,   0.944],
       [ -8.808,   4.569,  -4.539],
       [  8.334,   8.83 ,  10.718],
       [  3.563,  11.994,   3.638],
       [ -1.049,  -9.124,  -5.03 ],
       [-10.323, -14.502,  -5.596],
       [  9.55 , -13.084,  -9.938],
       [-14.504,   3.285,  11.252],
       [  8.46 , -13.347, -14.513],
       [ 11.509,   7.326,   9.999],
       [ 13.482,   6.181,  10.9  ],
       [-11.169,  -2.94 ,  -9.125],
       [  3.285, -14.032,  13.848],
       [ -0.6  ,  13.185, -11.535],
       [ 13.146, -12.501,  -1.605],
       [ 12.038,  11.56 ,   3.104],
       [ -4.797,   4.331,  12.748],
       [  3.005,  -6.883,  11.206],
       [ 14.361,  -6.807,   3.369],
       [  1.743,  -5.381,  -4.578],
       [  7.54 , -14.891,   7.293]])



def Beta(a,m):
    B = np.zeros((3,m))
    B[:,:a] = BETA_coeffs[:a].T
    return B
bias = np.zeros(shape=(3,))


from Cat_Select_101._utils import generate_Categorical
def Simulation_data(t,n_samples=sample_size,n_features=n_features,n_active=n_active,seed=0):
    X_Full = generate_Categorical.simulate_X(index=range(n_samples),n_col=(n_features[-1],0),generator=RNG(t+seed))
    Data = {f'm_{m}':{} for m in n_features}
    for m in n_features:
        X = X_Full.iloc[:,:m]
        for a in n_active[m]:
            y = generate_Categorical.simulate_catY(X,
                                                   Beta(a,m),use_intercept=bias,
                                                   classes=class_labels,generator=RNG(seed+t+a+m))
            Data[f'm_{m}'][f'a_{a}'] = (X,y)
    return Data


DATA = Parallel_itr(lambda t : Simulation_data(t,seed=0))
dump_to_file(DATA,'DATA')
test_DATA = Parallel_itr(lambda t : Simulation_data(t,seed=1))
dump_to_file(test_DATA,'test_DATA')


## ============================================================================

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty=None)


X,y = DATA[6]['m_100']['a_10']
X,y = X.iloc[:256,:],y.iloc[:256]
X_,y_ = test_DATA[0]['m_100']['a_10']
np.unique(y,return_counts=True)
generate_Categorical._softmax(X.to_numpy(),Beta(10,100),np.zeros(3))


import pandas as pd
S=pd.Series(index=[4,6,8,10,15,25,40,60,80,100])

model.fit(X,y)
S[100]=model.score(X_,y_)

model.fit(X.iloc[:,:10],y)
S[10]=model.score(X_.iloc[:,:10],y_)

model.fit(X.iloc[:,:15],y)
S[15]=model.score(X_.iloc[:,:15],y_)

model.fit(X.iloc[:,:25],y)
S[25]=model.score(X_.iloc[:,:25],y_)

model.fit(X.iloc[:,:40],y)
S[40]=model.score(X_.iloc[:,:40],y_)

model.fit(X.iloc[:,:60],y)
S[60]=model.score(X_.iloc[:,:60],y_)

model.fit(X.iloc[:,:80],y)
S[80]=model.score(X_.iloc[:,:80],y_)

model.fit(X.iloc[:,:4],y)
S[4]=model.score(X_.iloc[:,:4],y_)

model.fit(X.iloc[:,:6],y)
S[6]=model.score(X_.iloc[:,:6],y_)

model.fit(X.iloc[:,:8],y)
S[8]=model.score(X_.iloc[:,:8],y_)


## ============================================================================

T = pd.DataFrame()
T['rep1'] = S



