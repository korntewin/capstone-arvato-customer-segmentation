# builtin lib
import typing as t

# third party lib
import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
from scipy.sparse.construct import rand

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline


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


if __name__ == '__main__':
    
    test = np.array([[1, 2, 3], [4, 5, 6], [67, 870, 979], [6789, 98789, 678]])
    pca = PCA(random_state=42)

    print(calculate_appropriate_PCA_n_comp(pca=pca, X=test))

