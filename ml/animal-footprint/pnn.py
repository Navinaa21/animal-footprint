import numpy as np

class PNN:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        self.features = []
        self.labels = []

    def fit(self, features, labels):
        self.features = features
        self.labels = labels

    def predict(self, x):
        probs = []
        for i, f in enumerate(self.features):
            distance = np.linalg.norm(x - f)
            prob = np.exp(-distance**2 / (2 * self.sigma**2))
            probs.append((prob, self.labels[i]))

        probs.sort(reverse=True, key=lambda x: x[0])
        return probs[0][1]
