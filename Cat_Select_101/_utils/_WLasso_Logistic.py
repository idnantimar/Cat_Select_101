"""
Created on Wed Jan  3 18:09:03 2024

Topic: Weighted Lasso ('l1'-penalty, but different weights attached to different coefficients) implementation.
       For multi-class classification case.

NOTE : In the case of linear regression (or binary classification), when `coef_` is of dimension
(`n_features_in_`) Weighed LASSO can be solved with existing ``ElasticNet(l1_ratio=1)`` or
(``LogisticRegression(penalty='l1')``) in ``sklearn``, by just reweighting the columns of ``X`` accordingly. For
Multinomial Logistic Regression setup that is not possible, since
`coef_` is of shape (`n_classes_`,`n_features_in_`), coefficients in same column but different
rows can have different weightings.

So here we try a simple implementation of the problem using 'Generalized BLasso' approach.

References
----------
..[1] Zhao, Peng, and Bin Yu. "Stagewise lasso." The Journal of Machine Learning Research 8
(2007): 2701-2726.


@author: R.Nandi
"""



#### Logistic Regression + Weighted LASSO penalty =============================

from sklearn.base import BaseEstimator
import numpy as np



class _WLasso_Logistic(BaseEstimator):
    """
        In Weighted Lasso , instead of simply taking 'l1'-norm of model `coef_` as penalty,
        first we multiply each `coef_` with some known weights and then take the 'l1'-norm of
        the product as our penalty term.

        Note: Lasso can be considered as a special case of Weighted Lasso, where
        each `coef_` has weight 1 .

        USE CASE: When we have some prior information about the importance of the features,
        we can attach smaller penalty to more important features using this method.

        Parameters
        ----------
        loss_fun : a callable function of the type ``function(X,y,coef_,**kwargs) -> float``
            The unpenalized loss function to be minimized.
            Weighted Lasso penalty will be added implicitly.

        epsilon : float ; default 0.001
            The step-size of coordinate descent algorithm.
            Smaller step-size gives smoother regularization path, but needs more iterations to terminate.

        zeta : float ; default 1e-10
            For avoiding some round-off error only. Should be much smaller than `epsilon`.

    """
    def __init__(self,loss_fun,*,
                 epsilon=0.001,zeta=1e-10):
        self.loss_fun = loss_fun
        self.epsilon = epsilon
        self.zeta = zeta



    def fit(self,X,y,penalty_weights,*,max_iter=1000,compute_path=False,**kwargs):
        """
        The ``fit`` method of Weighted Lasso.

        Parameters
        ----------
        X : 2-D array of shape (n_samples,n_features)
            The training input samples.

        y : array of shape (n_samples,) or (n_samples,n_classes)
            The target values.

        penalty_weights : array of same shape as the `coef_` of underlying model.
            The weights of the 'l1'-penalties.

        max_iter : int ; default 1000
            The maximum number of iterations.

        compute_path : bool ; default False
            If True, fitted ``_WLasso_Logistic`` incstance will have an attribute `blasso_path_`.
            It can slow-down the fitting process slightly, since it involves copying data at each iteration.

        **kwargs : keyword arguments to be passed to `loss_fun`


        Attributes
        ----------
        history_ : dict
            {'penalized_loss': array of shape (`itr_`,) ; The penalized loss in each iteration ,

             'Cs': array of shape (`itr_`,) ; The regularization strength used in each iteration.
             If the last entry is -ve it indicates termination ,

             'penalty': array of shape (`itr_`,) ; The penalty in each iteration ,

             'loss': array of shape (`itr_`,) ; The unpenalized loss in each iteration
             }

        itr_ : int
            Number of iterations, including the initial iteration.

        blasso_path_ : dict ; Not available if `compute_path`=False
            Contains flattened array of `coef_` at each unique value of penalty.


        Returns
        -------
        An array of same shape as `coef_`.
            The fitted coefficients.

        """
        coef_shape = penalty_weights.shape
        all_indices = np.ndindex(coef_shape)
        eps = self.epsilon
        zeta = self.zeta
        def eps_Eij(i,j):
            ## 2-D array with eps at (i,j) ,
            ## 0 otherwise
            temp = np.zeros(coef_shape)
            temp[i,j] = eps
            return temp

        wlasso_penalty = lambda W : np.sum(penalty_weights*np.abs(W**2))
        loss_fun = lambda W : self.loss_fun(X,y,W,**kwargs)
        penalized_loss_function = lambda W,c : loss_fun(W) + c*wlasso_penalty(W)

        ### to store the values of loss , penalty temporarily .....
        temp_positive = np.zeros(coef_shape)
        temp_negative = np.zeros(coef_shape)
        ### to store the values of regularization strength .....
        Cs = np.zeros((max_iter+1,))
        ### store regularization path .....
        if compute_path : self.blasso_path_ = {}
        ### store loss and penalty values over the iterations .....
        penalized_loss_history = np.zeros((max_iter+1,))
        penalty_history = np.zeros((max_iter+1,))


        ### Initialization of BLasso
        #   ------------------------
        itr = 0
        ## initializing all coef_ at 0 .....
        current_state = np.ones_like(penalty_weights,order='C',dtype=float)
        initial_loss = loss_fun(current_state)
        initial_penalty = 0.
        if compute_path :
            self.blasso_path_[initial_penalty] = np.ravel(current_state.copy())
        ## finding steepest coordinate descent direction of loss .....
        for i,j in all_indices:
            epsEij = eps_Eij(i,j)
            temp_positive[i,j] = loss_fun(current_state + epsEij)
            temp_negative[i,j] = loss_fun(current_state - epsEij)
        ## stepping towards steepest descent direction .....
        if np.min(temp_negative) < np.min(temp_positive) :
            min_index = np.unravel_index(np.argmin(temp_negative),
                                         shape=coef_shape)
            current_state[min_index] -= eps
        else :
            min_index = np.unravel_index(np.argmin(temp_positive),
                                         shape=coef_shape)
            current_state[min_index] += eps
        ## finding initial value of regularization strength .....
        current_loss = loss_fun(current_state)
        current_penalty = wlasso_penalty(current_state)
        Cs[0] = (initial_loss - current_loss)/(current_penalty-initial_penalty)
        penalty_history[0] = current_penalty
        if compute_path :
            self.blasso_path_[current_penalty] = np.ravel(current_state.copy())
        penalized_loss_history[0] = current_loss + Cs[0]*current_penalty


        ### Iterative Updates
        #   -----------------
        c = Cs[0]
        while (itr<max_iter and c>0) :
            itr += 1
            ## finding steepest coordinate descent direction of penalized loss .....
            for i,j in all_indices:
                epsEij = eps_Eij(i,j)
                temp_positive[i,j] = penalized_loss_function(current_state + epsEij, c)
                temp_negative[i,j] = penalized_loss_function(current_state - epsEij, c)
            ## proposing a step towards steepest descent direction .....
            proposed_state = current_state.copy()
            if np.min(temp_negative) < np.min(temp_positive) :
                min_index = np.unravel_index(np.argmin(temp_negative),
                                             shape=coef_shape)
                proposed_state[min_index] -= eps
            else :
                min_index = np.unravel_index(np.argmin(temp_positive),
                                             shape=coef_shape)
                proposed_state[min_index] += eps
            ## possible Case-1 : penalized loss actually improved in proposed direction
            ##          Action : take a step at proposed direction, keep regularization strength fixed
            ## possible Case-2 : penalized loss minimization has reached a saturation, for given regularization strength
            ##          Action : relax regularization strength, by one unpenalized forward step
            penalized_loss_at_proposed_state = penalized_loss_function(proposed_state,c)
            if (penalized_loss_at_proposed_state <
                penalized_loss_history[itr-1] - zeta) : # [Case-1]
                # [[Action]]
                current_state = proposed_state
                Cs[itr] = c
                current_penalty = wlasso_penalty(current_state)
                penalty_history[itr] = current_penalty
                penalized_loss_history[itr] = penalized_loss_at_proposed_state
            else : # [Case-2]
                # [[Action]]
                # finding steepest coordinate descent direction of loss ...
                for i,j in all_indices:
                    epsEij = eps_Eij(i,j)
                    temp_positive[i,j] = loss_fun(current_state + epsEij)
                    temp_negative[i,j] = loss_fun(current_state - epsEij)
                # stepping towards steepest descent direction ...
                previous_penalty = penalty_history[itr-1]
                previous_loss = penalized_loss_history[itr-1] - c*previous_penalty
                if np.min(temp_negative) < np.min(temp_positive) :
                    min_index = np.unravel_index(np.argmin(temp_negative),
                                                 shape=coef_shape)
                    current_state[min_index] -= eps
                else :
                    min_index = np.unravel_index(np.argmin(temp_positive),
                                                 shape=coef_shape)
                    current_state[min_index] += eps
                # update ...
                current_loss = loss_fun(current_state)
                current_penalty = wlasso_penalty(current_state)
                c = min(c,(previous_loss - current_loss)/(current_penalty-previous_penalty))
                penalty_history[itr] = current_penalty
                Cs[itr] = c
                penalized_loss_history[itr] = current_loss + c*current_penalty
            ## blasso path .....
            if compute_path :
                self.blasso_path_[current_penalty] = np.ravel(current_state.copy())
            ## ..... ..... .....


        ### count of iteration including the 0-th iteration
        #   -----------------------------------------------
        itr += 1


        ### Outputs
        #   -------
        self.history_ = {'penalized loss': penalized_loss_history[:itr],
                         'Cs':Cs[:itr],
                         'penalty':penalty_history[:itr],
                         'loss': penalized_loss_history[:itr] - Cs[:itr]*penalty_history[:itr]}
        self.itr_ = itr
        return current_state




#### ==========================================================================
