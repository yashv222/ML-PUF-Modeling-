import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from itertools import combinations

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################
    
    # Use this method to train your models using training CRPs
    # X_train has 8 columns containing the challenge bits
    # y_train contains the values for responses
    
    # THE RETURNED MODEL SHOULD BE ONE VECTOR AND ONE BIAS TERM
    # If you do not wish to use a bias term, set it to 0

    X_train_mapped = my_map(X_train)
    model = LinearSVC(C=10, max_iter=1000)
    # model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, fit_intercept=False, C = 100.0, random_state=42)
    model.fit(X_train_mapped, y_train)
    w = model.coef_.flatten()
    b = 0.0
    return w, b

################################
# Non Editable Region Starting #
################################
def my_map(X):
################################
#  Non Editable Region Ending  #
################################

    X_bin = 1 - 2 * X  # Convert {0,1} to {+1,-1}
    m, n = X_bin.shape  # Expecting n = 8

    assert n == 8, "Expected 8 input features for c_0 to c_7"

    feat_list = []

    # For k = 1 to 8, generate all k-wise products of columns
    for k in range(1, n + 1):
        for idxs in combinations(range(n), k):
            # Multiply across selected columns
            prod = np.prod(X_bin[:, idxs], axis=1, keepdims=True)
            feat_list.append(prod)

    feat = np.hstack(feat_list)  # Final shape: (m, 255)
    return feat

################################
# Non Editable Region Starting #
################################
def my_decode(w):
################################
#  Non Editable Region Ending  #
################################
    # Randomized version to generate any valid (non-negative) delays
    # that still reconstruct the original linear model exactly

    # Extract bias and weights
    b = w[-1]
    w_vector = np.array(w[:-1])

    # Initialize alpha and beta
    alpha = np.zeros(64)
    beta = np.zeros(64)
    alpha[0] = w_vector[0]
    for i in range(1, 64):
        alpha[i] = w_vector[i]  # w_i = alpha_i + beta_{i-1}
        beta[i - 1] = w_vector[i] - alpha[i]
    beta[63] = b  # last beta from bias

    # Derive D_i and E_i for all 0 <= i <= 63
    D = alpha + beta  # D_i = p_i - q_i
    E = alpha - beta  # E_i = r_i - s_i

    # Initialize output delays
    p = np.zeros(64)
    q = np.zeros(64)
    r = np.zeros(64)
    s = np.zeros(64)

    # Randomized solution: instead of always minimizing total delay,
    # randomly generate valid values satisfying the constraints.
    for i in range(64):
        # Choose a random shift ≥ max(-D[i], 0)
        delta_pq = np.random.uniform(low=max(-D[i], 0), high=max(1.0, abs(D[i])) + 1.0)
        p[i] = D[i] + delta_pq
        q[i] = delta_pq

        delta_rs = np.random.uniform(low=max(-E[i], 0), high=max(1.0, abs(E[i])) + 1.0)
        r[i] = E[i] + delta_rs
        s[i] = delta_rs

    return p, q, r, s
