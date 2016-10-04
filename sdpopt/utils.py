from __future__ import print_function, division, absolute_import

import json
import numpy as np
from sklearn.utils import check_random_state
from sklearn.externals.joblib import load, dump
from numpy.linalg import norm

##########################################################################
# MSLDS Utils (experimental)
##########################################################################


def iter_vars(A, Q, N):
    """Utility function used to solve fixed point equation
       Q + A D A.T = D
       for D
     """
    V = np.eye(np.shape(A)[0])
    for i in range(N):
        V = Q + np.dot(A, np.dot(V, A.T))
    return V

##########################################################################
# END of MSLDS Utils (experimental)
##########################################################################

# TODO: FIX THIS!
def compute_eigenspectra(self):
    """
    Compute the eigenspectra of operators A_i
    """
    eigenspectra = np.zeros((self.n_states,
                            self.n_features, self.n_features))
    for k in range(self.n_states):
        eigenspectra[k] = np.diag(np.linalg.eigvals(self.As_[k]))
    return eigenspectra

# TODO: FIX THIS!
def load_from_json_dict(model, model_dict):
    # Check that the num of states and features agrees
    n_features = float(model_dict['n_features'])
    n_states = float(model_dict['n_states'])
    if n_features != self.n_features or n_states != self.n_states:
        raise ValueError('Invalid number of states or features')
    # read array values from the json dictionary
    Qs = []
    for Q in model_dict['Qs']:
        Qs.append(np.array(Q))
    As = []
    for A in model_dict['As']:
        As.append(np.array(A))
    bs = []
    for b in model_dict['bs']:
        bs.append(np.array(b))
    means = []
    for mean in model_dict['means']:
        means.append(np.array(mean))
    covars = []
    for covar in model_dict['covars']:
        covars.append(np.array(covar))
    # Transmat
    transmat = np.array(model_dict['transmat'])
    # Create the MSLDS model
    self.Qs_ = Qs
    self.As_ = As
    self.bs_ = bs
    self.means_ = means
    self.covars_ = covars
    self.transmat_ = transmat
