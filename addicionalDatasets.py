import numpy as np
from sklearn import datasets

def make_anisotropic_blobs(n_samples=500, random_state=170):
        # Generates standard blobs, then stretches them
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        return X_aniso, y

def make_random_state(n_samples=500, random_state=30):
        rng = np.random.RandomState(random_state)
        return rng.rand(n_samples, 2), None