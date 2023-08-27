import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from archetypes_analysis import get_archetypes
import matplotlib.pyplot as plt
import archetypes as arch
from sklearn.model_selection import train_test_split


# step 1 - PCA 
# Principal Component Analysis to find 8 principal components
# TODO: Mouayad and Natasha

    # ------ implement it here ------
def perform_PCA(data, n_components=8):
    data = data.select_dtypes(include=[np.number])
    pca = PCA(n_components=8, random_state=2023)
    X = data.values
    pca.fit(X)
    data_scaled = pca.transform(X)

    print(data_scaled.shape)
    print(sum(pca.explained_variance_ratio_ * 100))

    return data_scaled


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

def residual_sum_of_squares(X):
    # ------ implement it here ------
    res = []
    archetype_range = range(1, 21)
    for n_archetypes in archetype_range:
        # aa = arch.Archetypes(n_archetypes=n_archetypes)
        aa = arch.AA(n_archetypes=n_archetypes)
        aa.fit(X)
        X_trans = aa.transform(X)

        reconstructed = np.dot(X_trans, aa.archetypes_)
        residuals = X - reconstructed
        print(X_trans.shape)
        print(aa.archetypes_)

        res.append(np.sum(residuals ** 2))

    plt.plot(archetype_range, res)
    plt.xlabel('Number of Archetypes')
    plt.ylabel('Residual Sum of Squares')
    plt.title('RSS Curve for Different Number of Archetypes')
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------


# step 4 - Use elbow method or silhouette method to find most appropriate archetypal model.
# TODO: All of us


# ------------------------------------------------------------------------------------------------------------------------------------------------------------


# step 5 - repeat the above,except, before starting, hold out 20% of the data. 
# This will be our validation set. Train the model on the 80% that remains. 
# See how the model performs reconstructing the test data. 

def test_model_performance(X):
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=2023)
    res_train = []
    res_val = []
    archetype_range = range(1, 21)
    for n_archetypes in archetype_range:
        aa = arch.AA(n_archetypes=n_archetypes)
        aa.fit(X_train)
        
        # Reconstruct training data
        X_trans_train = aa.transform(X_train)
        reconstructed_train = np.dot(X_trans_train, aa.archetypes_)
        residuals_train = X_train - reconstructed_train
        res_train.append(np.sum(residuals_train ** 2))
        
        # Reconstruct validation data
        X_trans_val = aa.transform(X_val)
        reconstructed_val = np.dot(X_trans_val, aa.archetypes_)
        residuals_val = X_val - reconstructed_val
        res_val.append(np.sum(residuals_val ** 2))

    plt.plot(archetype_range, res_train, label='Train')
    plt.plot(archetype_range, res_val, label='Validation')
    plt.xlabel('Number of Archetypes')
    plt.ylabel('Residual Sum of Squares')
    plt.title('RSS Curve for Different Number of Archetypes')
    plt.legend()
    plt.show()


# main function

if __name__ == "__main__":
    data = pd.read_csv("dataset.csv")
    # 448 columns / dimensions -- reduce this to 8

    data_scaled = perform_PCA(data)

    res = test_model_performance(data_scaled)
    # res = residual_sum_of_squares(data_scaled)
    # res = get_archetypes(data_scaled)

