# custom_naive_bayes.py

import numpy as np

class CustomNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for cls in self.classes:
            X_cls = X[y == cls]
            self.mean[cls] = np.mean(X_cls, axis=0)
            self.var[cls] = np.var(X_cls, axis=0)
            self.priors[cls] = X_cls.shape[0] / X.shape[0]

    def gaussian_density(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        posteriors = []

        for x in X:
            class_posteriors = []

            for cls in self.classes:
                prior = np.log(self.priors[cls])
                conditional = np.sum(np.log(self.gaussian_density(cls, x)))
                posterior = prior + conditional
                class_posteriors.append(posterior)

            posteriors.append(self.classes[np.argmax(class_posteriors)])

        return posteriors
