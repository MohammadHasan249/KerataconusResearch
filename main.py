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
    pass


# step 2 - Archetypal Analysis
# Archetypal analysis on these principal components using python package 'archetypes'.
# TODO: Mohammad

# I'm doing this in another file (`archetypes_analysis.py`) and will import it here.



# step 3 - Create a residual sum of squares curve for all archetypes.
# This will be based off how well the various different archetypal models reconstruct the data. 
# I would suggest doing this from 1 - 20 archetypes or something initially
# (we won't know what works until you have found the archetypes - there is a bit of trial and error here).
# TODO: Mohammad

def residual_sum_of_squares(archetypes):
    # ------ implement it here ------
    pass


# step 4 - Use elbow method or silhouette method to find most appropriate archetypal model.
# TODO: All of us


# step 5 - repeat the above,except, before starting, hold out 20% of the data. 
# This will be our validation set. Train the model on the 80% that remains. 
# See how the model performs reconstructing the test data. 







