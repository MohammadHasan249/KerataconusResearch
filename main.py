import pandas as pd
import sklearn
import numpy as np
import archetypes as arch
from archetypes_analysis import get_archetypes

X = np.random.normal(0, 1, (100, 2))

aa = arch.AA(n_archetypes=4)

X_trans = aa.fit_transform(X)
print(X_trans.shape)

data = pd.read_csv("dataset.csv")

# step 1 - PCA 
# Principal Component Analysis to find 8 principal components
# TODO: Mouayad and Natasha

def PCA():
    # ------ implement it here ------
def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as NP
    from scipy import linalg as LA
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs

def test_PCA(data, dims_rescaled_data=2):
    '''
    test by attempting to recover original data array from
    the eigenvectors of its covariance matrix & comparing that
    'recovered' array with the original data
    '''
    _ , _ , eigenvectors = PCA(data, dim_rescaled_data=2)
    data_recovered = NP.dot(eigenvectors, m).T
    data_recovered += data_recovered.mean(axis=0)
    assert NP.allclose(data, data_recovered)
    

def plot_pca(data):
    from matplotlib import pyplot as MPL
    clr1 =  '#2026B2'
    fig = MPL.figure()
    ax1 = fig.add_subplot(111)
    data_resc, data_orig = PCA(data)
    ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr1, mec=clr1)
    MPL.show()





>>> df = "~/keratoconus dataset.csv"
>>> data = NP.loadtxt(df, delimiter=',')
>>> # remove class labels
>>> data = data[:,:-1]
>>> plot_pca(data)




    pass


-----------------------------------------------------------------------------------------------------------------------------------------------------------


# step 2 - Archetypal Analysis
# Archetypal analysis on these principal components using python package 'archetypes'.
# TODO: Mohammad

# I'm doing this in another file (`archetypes_analysis.py`) and will import it here.


-----------------------------------------------------------------------------------------------------------------------------------------------------------


# step 3 - Create a residual sum of squares curve for all archetypes.
# This will be based off how well the various different archetypal models reconstruct the data. 
# I would suggest doing this from 1 - 20 archetypes or something initially
# (we won't know what works until you have found the archetypes - there is a bit of trial and error here).
# TODO: Mohammad

def residual_sum_of_squares(archetypes):
    # ------ implement it here ------
    pass

----------------------------------------------------------------------------------------------------------------------------------------------------------


# step 4 - Use elbow method or silhouette method to find most appropriate archetypal model.
# TODO: All of us


------------------------------------------------------------------------------------------------------------------------------------------------------------


# step 5 - repeat the above,except, before starting, hold out 20% of the data. 
# This will be our validation set. Train the model on the 80% that remains. 
# See how the model performs reconstructing the test data. 







