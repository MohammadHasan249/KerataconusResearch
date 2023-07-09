import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import numpy as np
from archetypes_analysis import get_archetypes
import matplotlib as plt


data = pd.read_csv("dataset.csv")
# 448 columns / dimensions -- reduce this to 8
# 80.49% of our variance
X = data.values

pca = PCA(n_components=8, random_state=2023)
pca.fit(X)
X_scaled = pca.transform(X)

print(X_scaled.shape)
print(sum(pca.explained_variance_ratio_ * 100))


# step 1 - PCA 
# Principal Component Analysis to find 8 principal components
# TODO: Mouayad and Natasha

    # ------ implement it here ------
def perform_PCA(data, n_components=8):
    data = data.select_dtypes(include=[np.number])
    pca = PCA(n_components=8, random_state=2023)
    pca.fit(X)
    X_scaled = pca.transform(X)

    print(X_scaled.shape)
    print(sum(pca.explained_variance_ratio_ * 100))

    return X_scaled

def plot_pca(data):
    from matplotlib import pyplot as MPL
    clr1 =  '#2026B2'
    fig = MPL.figure()
    ax1 = fig.add_subplot(111)
    data_resc, data_orig = PCA(data)
    ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr1, mec=clr1)
    MPL.show()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------

# step 2 - Archetypal Analysis
# Archetypal analysis on these principal components using python package 'archetypes'.
# TODO: Mohammad

# I'm doing this in another file (`archetypes_analysis.py`) and will import it here.



# -----------------------------------------------------------------------------------------------------------------------------------------------------------


# step 3 - Create a residual sum of squares curve for all archetypes.
# This will be based off how well the various different archetypal models reconstruct the data. 
# I would suggest doing this from 1 - 20 archetypes or something initially
# (we won't know what works until you have found the archetypes - there is a bit of trial and error here).
# TODO: Mohammad

def residual_sum_of_squares(archetypes):
    # ------ implement it here ------
    pass

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


# step 4 - Use elbow method or silhouette method to find most appropriate archetypal model.
# TODO: All of us


# ------------------------------------------------------------------------------------------------------------------------------------------------------------


# step 5 - repeat the above,except, before starting, hold out 20% of the data. 
# This will be our validation set. Train the model on the 80% that remains. 
# See how the model performs reconstructing the test data. 







