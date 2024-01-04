"""
Created on Sun Dec 31 23:36:20 2023

Topic: An implementation of Logistic Regression with any kind of penalty function.

@author: R.Nandi
"""


#### Implementing Logistic Regression using tensorflow & sklearn ==============

import tensorflow as tf
from tensorflow import keras
from sklearn.base import BaseEstimator



class _custom_LogisticReg(BaseEstimator):
    """
        We know that, constructing a single-layer neural network with 'softmax' activation and
        minimizing 'categorical_crossentrophy' is conceptually equivalent to fitting Logistic
        Regression with maximum likelihood estimation (MLE).

        [ For internal use mainly ]

        This class determines the architecture of that network.
        Override the ``fit`` and ``score`` method when necessary.

        Note: The network returns ``logits`` instead of ``probability``.

    """
    def __init__(self,n_features_in,n_classes,*,
                 penalty_class,penalty_param,compile_configuration,
                 random_state,dtype):
        self.n_features_in = n_features_in
            ## input dimension
        self.n_classes = n_classes
            ## output dimension
        self.penalty_class = penalty_class
            ## the class of required penalty function, not an instance
        self.penalty_param = penalty_param
            ## this can be anything, like- scalar,array,list etc.
             ## design your custom penalty accordingly , in terms of a single tuning parameter.
            ## its best possible value will be decided by GridSearch later
        self.compile_configuration = compile_configuration
            ## dict containing optimizer and metrics
        self.random_state = random_state
        self.dtype = dtype

    def _build(self,W0,b0):
        _penalty_fun = self.penalty_class(self.penalty_param,self.dtype)
            ## a callable penalty function,
             ## derived from the penalty_class with given penalty_param
        _kernel_initializer = _custom_initializer(W0)
        _bias_initializer = _custom_initializer(b0)
            ## pass the coef_.T and intercept_ of a LogisticRegression(penalty=None)
             ## to leverage the training process
        nn = keras.Sequential(name='Logistic_Regression')
        nn.add(keras.Input(shape=(self.n_features_in,),dtype=self.dtype,name="Input"))
        nn.add(keras.layers.Dense(units=self.n_classes,name='SoftMax',
                                  use_bias=True,
                                  kernel_initializer=_kernel_initializer,
                                  bias_initializer=_bias_initializer,
                                  kernel_regularizer=_penalty_fun,
                                  dtype=self.dtype))
        self.compile_configuration.update({'loss':keras.losses.CategoricalCrossentropy(from_logits=True,
                                                    label_smoothing=0.0,
                                                    reduction="sum_over_batch_size")})
        nn.compile(**self.compile_configuration)
        self.nn = nn

    def fit(self,X,y,weights0,bias0,**kwargs):
        self._build(weights0,bias0)
        self.nn.fit(X,y,**kwargs)

    def score(self,X,y):
        return self.nn.evaluate(X,y,verbose=0)[1]


#### ==========================================================================

##> ......................................................................

class _custom_initializer(keras.initializers.Initializer):
    """
        For leveraging the training by providing initial guess about weights

        Initial guess must be in required shape -
            for kernel weights shape=(input_dim,output_dim)
            for bias shape=(n_units,)
    """
    def __init__(self,initial_guess):
        self.initial_guess = initial_guess
    def __call__(self,shape,dtype):
        return tf.Variable(self.initial_guess,dtype=dtype)
    def get_config(self):
        return {'initial_guess':self.initial_guess}

#> .......................................................................
