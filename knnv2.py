import numpy as np
from sklearn import datasets as ds, model_selection as ms, neighbors as n
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances as pdistss
from collections import Counter
import pandas as pd


class KNNClassif:
    
    def __init__(self, k=3, metric = "euclid", p=3):
        
        self.k = k
        self.metric = metric
        self.p = p
        
    
    def fit(self, X_train, y_train):
        
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        
    def _get_dist(self, x_test):
        if self.metric == "euclid":
            return np.sqrt(np.sum((x_test - self.X_train)**2, axis=1))
        elif self.metric == "manhatt":
            return np.sum(np.abs(x_test - self.X_train), axis=1)
        elif self.metric == "chebysh":
            return np.max(np.abs(x_test - self.X_train), axis=1)
        elif self.metric == "minovsk":
            return np.sum(np.abs(x_test - self.X_train) ** self.p, axis=1) ** (
                1 / self.p
            )
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _predict_one(self, x_test):
        
        dist = self._get_dist(x_test)
        
        k_i = np.argsort(dist)[:self.k]
        k_elem = self.y_train[k_i]
        
        return Counter(k_elem).most_common(1)[0][0]
    
    def predict(self, X_test):
        
        return np.array([self._predict_one(i) for i in X_test])
    
    
    def get_accuracy(self, X_test, y_test):
        
        pred = self.predict(X_test)
        
        return np.mean(pred == y_test)
        


def analysys(dataset):
    
    X = dataset.data
    y = dataset.target
    
    X_train, X_test, y_train, y_test = ms.train_test_split(
    X,
    y, 
    test_size=0.3, 
    shuffle=True, 
    random_state=52
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    results = []
    
    for k in range(3,11):
        for metr in metrics:
            my_knn = KNNClassif(k=k, metric=metr[1])
            my_knn.fit(X_train, y_train)
            my_acc = my_knn.get_accuracy(X_test, y_test)
            
            skl_knn = n.KNeighborsClassifier(n_neighbors=k, metric=metr[2], p=3)
            skl_knn.fit(X_train, y_train)
            skl_acc =skl_knn.score(X_test, y_test)
            
            results.append((k, metr[0], round(my_acc, 4), round(skl_acc, 4)))

    df = pd.DataFrame(results, columns= ("k", "metric", "my_accuracy", "skl_accuracy"))
    pivot_my = df.pivot_table(index="k", columns="metric", values="my_accuracy")
    pivot_skl = df.pivot_table(index="k", columns="metric", values="skl_accuracy")
    best_my = df.loc[df['my_accuracy'].idxmax()]
    best_skl = df.loc[df['skl_accuracy'].idxmax()]
    
    return df, pivot_my, pivot_skl, best_my, best_skl

def analysis_all(datasets):
    
    results = {}
    
    for name, dataset in datasets.items():
        print(name, "\n")
        df, pivot_my, pivot_skl, best_my, best_skl = analysys(dataset)
        results.update({
            name: (df, pivot_my, pivot_skl, best_my, best_skl)
        })
        print(pivot_my, "\n")
        print(f'лучший результат: \n{best_my}\n')
        print(pivot_skl, "\n")
        print(f'лучший результат: \n{best_skl}\n')
    
    return results


def analysys_rad(dataset):
    
    X = dataset.data
    y = dataset.target
        
    X_train, X_test, y_train, y_test = ms.train_test_split(
    X,
    y, 
    test_size=0.3, 
    shuffle=True, 
    random_state=52
    )
        
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    distss= pdistss(X_train)
    np.fill_diagonal(distss, np.inf)
    nn_dist = distss.min(axis=1)
    med = np.median(nn_dist)
    
    radss = [round((med * n), 3) for n in [1, 2, 3, 4, 6, 10, 12, 14, 16, 20, 30]]

    results = []
    
    for r in radss:
        radnn = n.RadiusNeighborsClassifier(radius=r, outlier_label="most_frequent")
        radnn.fit(X_train, y_train)
        rad_acc = radnn.score(X_test, y_test)
        results.append((r, round(rad_acc, 4)))
    
    df = pd.DataFrame(results, columns=("r", "accuracy"))
    best = df.loc[df["accuracy"].idxmax()]
    
    return df, best


def analysys_rad_all(datasets):
    
    results = {}
    
    for name, dataset in datasets.items():
        print(name, "\n")
        df, best = analysys_rad(dataset)
        results.update({name: (df, best)})
        print(df, "\n")
        print(f'лучший результат: \n{best}\n')
    
    return results
        

metrics = [
    ("Евклидова", "euclid", "euclidean"), 
    ("Манхэттенская", "manhatt", "manhattan"), 
    ("Чебышёва", "chebysh", "chebyshev"), 
    ("Миновского", "minovsk", "minkowski"),
]

datasets = {
    "iris": ds.load_iris(),
    "digits": ds.load_digits(),
    "wine": ds.load_wine(),
    "breast_cancer": ds.load_breast_cancer()
}        


results = analysis_all(datasets)

results_rad = analysys_rad_all(datasets)