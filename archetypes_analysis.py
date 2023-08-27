import archetypes as arch

def get_archetypes(X):
    aa = arch.AA(n_archetypes=4)
    X_trans = aa.fit_transform(X)
    print(X_trans.shape)
    print(aa.archetypes_)



# X = np.random.normal(0, 1, (100, 2))

# aa = arch.AA(n_archetypes=4)

# X_trans = aa.fit_transform(X)
# print(X_trans.shape)