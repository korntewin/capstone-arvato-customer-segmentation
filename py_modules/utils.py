# builtin lib
import typing as t

# third party lib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import loguniform, uniform, randint

def calculate_appropriate_PCA_n_comp(*, pca: PCA, X: pd.DataFrame, threshold: float=0.9) -> t.Tuple[int, np.ndarray]:
    '''This function calculate the appropriate number of components
    that can retain 90% of explainable variance for a given PCA and dataframe

    Returns:
        n_components: number of components for PCA
    '''

    # fit the pca to calculate explain variance ratio
    pca.fit(X)

    var_ratio: np.ndarray = pca.explained_variance_ratio_
    cumsum_var_ratio: np.ndarray = np.cumsum(var_ratio)

    # identify the position that cumsum of var ratio
    # greater than threshold
    is_greater: np.ndarray = cumsum_var_ratio >= threshold

    n_comp: int = np.argmax(is_greater) + 1

    return n_comp, cumsum_var_ratio


def calculate_appropriate_n_cluster_KMeans(*, km: MiniBatchKMeans, \
    X: pd.DataFrame, n_range: range=range(2,15)) -> t.List[float]:
    '''Calculate appropriate n using elbow point method.
    However, in the first version, I will just only return the average distance
    of the KMeans to be plotted and select the n cluster manually
    '''

    # init return array
    avg_distances = [1.0 for _ in n_range]

    for i, n in enumerate(n_range):
        km.set_params(n_clusters=n)
        km.fit(X)

        inertia_: float = km.inertia_
        avg_distances[i] = inertia_

    return avg_distances


if __name__ == '__main__':
    
    test = np.array([[1, 2, 3], [4, 5, 6], [67, 870, 979], [6789, 98789, 678]])
    pca = PCA(random_state=42)
    print(calculate_appropriate_PCA_n_comp(pca=pca, X=test))

    km = MiniBatchKMeans(random_state=42)
    print(calculate_appropriate_n_cluster_KMeans(km=km, X=test, n_range=range(1,5)))

