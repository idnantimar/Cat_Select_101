"""
Created on Fri Jan  5 02:58:36 2024

Topic: Functions to simulate categorical data ``(X,y)``.

The ``make_classification`` available in ``sklearn.datasets`` -

(I) does not reveal the true coefficients,

(II) does not allow to customize the correlation between the features,

(III) does not provide options to include categorical features,

So, we will use these custom functions for simulation in this project.


@author: R.Nandi
"""



import numpy as np
import pandas as pd


#### Simulate Predictors ======================================================

##> ............................................................

def _gen_Normal(rows=20,cols=5,rng=np.random.default_rng()):
    mu = np.zeros((cols,))
    Sig = np.identity(cols)
    return rng.multivariate_normal(mu,Sig,size=rows)

def _gen_Category(rows=20,cols=3,rng=np.random.default_rng(),*,n_class=[3,2]):
    k = len(n_class)
    Out = np.full((rows,cols),fill_value=None)
    possible_categories = list(ascii_lowercase)
    one_col = lambda c : rng.choice(possible_categories[:c],
                                    size=rows,replace=True)
    for j in range(cols):
        Out[:,j] = one_col(n_class[j%k])
    return Out

#> ............................................................



def simulate_X(index=range(100),n_col=(5,0),*,
              N_type=(_gen_Normal,{}),
              C_type=(_gen_Category,{'n_class':[3,2]}),
              generator=np.random.default_rng(),
              feature_names=(None,None)):
    """
    A function to simulate a DataMatrix from a joint-distribution specified.

    Parameters
    ----------
    index : RangeIndex ; default ``range(100)``
        Index of the simulated data, also specifying the number of observations to be generated.

    n_col : tuple ; default (5,0)
        A tuple of the form (N,C) , where N is number of numeric columns and
        C is number of categorical columns. Put (N,0) if no categorical columns are needed.
        If present, categorical columns will be concatinated at the right end of numeric columns.

    N_type : (``function(rows,cols,rng,**kwargs)->array,kwargs``) to generate numeric columns
        Example1 [default choice] -
            >>> def gen_Normal(rows,cols,rng):
            ...:    mu = np.zeros((cols,))
            ...:    Sig = np.identity(cols,)
            ...:    return rng.multivariate_normal(mu,Sig,size=rows)

            >>> N_type = (gen_Normal,{})

        This will simulate independent normal features of specified size.

        Example2 -
            >>> def gen_Normal(rows,cols,rng):
            ...:    mu = np.zeros((cols,))
            ...:    Sig = (np.identity(cols,) +
            ...:           np.diag([0.5]*(cols-1),k=1) +
            ...:            np.diag([0.5]*(cols-1),k=-1))
            ...:    return rng.multivariate_normal(mu,Sig,size=rows)

            >>> N_type = (gen_Normal,{})

        This will simulate features of specified size from an AR(0.5) process.

    C_type : (``function(rows,cols,rng,**kwargs)->array,kwargs``) to generate categorical columns
        Example1 [default choice] -
            >>> def gen_Category(rows,cols,rng,*,n_class):
            ...:    k = len(n_class)
            ...:    Out = np.full((rows,cols),fill_value=None)
            ...:    possible_categories = list(ascii_lowercase)
            ...:    one_col = lambda c : rng.choice(possible_categories[:c],
            ...:                                size=rows,replace=True)
            ...:    for j in range(cols):
            ...:        Out[:,j] = one_col(n_class[j%k])
            ...:    return Out

            >>> C_type=(gen_Category,{'n_class':[3,2]})

        This will simulate independent categorical features of specified size, where
        1st one has 3-categories, 2nd one has 2-categories, 3rd one has 3-categories ... etc.

        Example2 -
            >>> def gen_Category(rows,cols,rng):
                ...:    Out = np.full((rows,cols),fill_value=None)
                ...:    one_col = lambda : rng.choice([0,1,2],
                ...:                            size=rows,replace=True)
                ...:    for j in range(cols):
                ...:        if not j%2:
                ...:            u = one_col()
                ...:            Out[:,j] = u
                ...:        else:
                ...:            Out[:,j] = (u+1)%3
                ...:    return Out

            >>> C_type=(gen_Category,{})

        This will simulate categorical features , where each odd column is Uniform({0,1,2}) ,
        and each even column has immediate next category of the preceeding odd column, following
        the cycle 0->1->2->0 .

    [ Note: simulated categorical columns will always be independent of numeric columns in this ``simulate_X`` function ]

    generator : random number generator ; default ``np.random.default_rng(seed=None)``

    feature_names : tuple (N_names,C_names) ; default (None,None)
        Where N_names and C_names are array of feature names , having shape
        (`n_col`[0],) and (`n_col`[1],) respectively. If (None,None) default names
        [N_0,N_1,...] and [C_0,C_1,...] will be attached.

    Returns
    -------
    DataFrame of shape (`n_obs`,``sum``(`n_col`))
        Here each row is an observation , each column is a feature

    """
    n_obs = len(index)
    n_Num,n_Cat = n_col
    N_names,C_names = feature_names
    ## simulating numeric data .....
    if n_Num>0 :
        N_gen,N_kwargs = N_type
        X_Num = N_gen(n_obs,n_Num,generator,**N_kwargs)
        if N_names is None :
            N_names = np.vectorize(lambda j : "N_"+str(j))(range(n_Num))
        X_Num = pd.DataFrame(X_Num,index=index,columns=N_names)
    else :
        X_Num = pd.DataFrame(index=index,dtype=float)
    ## simulating categorical data .....
    if n_Cat>0 :
        C_gen,C_kwargs = C_type
        X_Cat = C_gen(n_obs,n_Cat,generator,**C_kwargs)
        if C_names is None :
            C_names = np.vectorize(lambda j : "C_"+str(j))(range(n_Cat))
        X_Cat = pd.DataFrame(X_Cat,index=index,columns=C_names)
    else :
        X_Cat = pd.DataFrame(index=index,dtype=float)

    ## combined output .....
    return pd.concat([X_Num,X_Cat],axis=1)


#### ==========================================================================
##
###
####
###
##
#### Simulate Response|observed_Predictors ====================================


##> ............................................................

def _softmax(x,B):
    """
        *  return multinomial probabilities at observation x of shape (n_features,)
           or over a dataset of shape (n_samples,n_features)

        *  B is of the shape (n_classes,n_features) , the coefficients of the model

        Note: for each observation , returned probabilities sum to 1.
    """
    Bx = np.matmul(B,x.T)
    exp_Bx = np.exp(Bx - np.max(Bx,axis=None))
    out = exp_Bx / exp_Bx.sum(axis=0,keepdims=True)
        ## after x.T each column is an observation,
         ## we want to convert it back as rows
    return out.T

#> .............................................................


def simulate_catY(X,Beta_coeffs,*,
                  use_intercept = None,
                  Phi= lambda x: x, return_FeatureMap=False,
                  classes = None,
                  generator = np.random.default_rng()):
    """
    A function to simulate categorical response based on Multinomial conditional probability & an observed DataMatrix.

    Parameters
    ----------
    X : DataFrame of shape (n_samples,n_features)
        The observed Predictors.

    Beta_coeffs : 2D-array of shape (n_classes,dim_featuremap)
        The coefficients to be used to model the multinomial probabilities. Each row corresponds to a response category.

            *   Setting all the coeficients to 0 results in independence of Y & X

            *   Setting any column constant implies that corresponding feature has no role in the model

    use_intercept : array of shape (n_classes,) ; default None
        The intercepts to be used to model the multinomial probabilities. Each component
        corresponds to one class. None implies no-intercept model.

    Phi : a function that takes an array of shape (n_features,) as input,
    returns an array of shape (dim_featuremap,) as output ; default ``lambda x: x``
        The FeatureMap to be applied on each rows of X. Default choice is Linear FeatureMap.

    return_FeatureMap : bool ; default False
        Whether the Phi(X) of shape (n_samples,dim_featuremap) will be returned(True) or not(False) at the end.

    classes : array of shape (n_classes,)
        The list of response category names. If None, default names [cls_0,cls_1,cls_2,...]
        will be attached.

    generator : random number generator ; default ``np.random.default_rng(seed=None)``


    Returns
    -------
    * Series of shape (n_samples,) when return_FeatureMap=False
        Contains the name of the simulated category at every index.

    * (Y,phiX) when return_FeatureMap=True
        Y is the Series of response values , shape (n_samples,).
        phiX is the transformed version of X after applying given feature-map ,
        shape (n_samples,dim_featuremap).

    """
    X = pd.DataFrame(X).copy()
    Beta_coeffs = np.array(Beta_coeffs)
    k,m = Beta_coeffs.shape
    ## applying feature-map .....
    phiX = X.apply(Phi,axis=1,result_type='expand')
    ## linear transformation .....
    Bx = np.matmul(phiX,Beta_coeffs.T).to_numpy()
    ## adding intercept .....
    if use_intercept is None : use_intercept = np.zeros((1,k))
    Bx += use_intercept
    ## softmax transformation .....
    exp_Bx = np.exp(Bx - np.max(Bx,axis=None))
    Probs = exp_Bx / exp_Bx.sum(axis=1,keepdims=True)
    ## simulating Y .....
    Y = generator.multinomial(1,Probs).argmax(axis=1)
    Y = pd.Series(Y,index=X.index)
    if classes is None : classes = [('cls_'+str(i)) for i in range(k)]
    Y = Y.apply(lambda i: classes[i])

    ## output .....
    return Y if not return_FeatureMap else (Y,phiX)




#### ==========================================================================
