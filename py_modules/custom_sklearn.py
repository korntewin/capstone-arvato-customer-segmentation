import typing as t

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

class NominalEncoderAndMinMaxScaler(BaseEstimator, TransformerMixin):
    '''Wrap onehot encoder to encode nomincal encoder on dataframe
    while preserving column name'''

    def __init__(self, nominal_col: t.List[str], non_nominal_col: t.List[str]):
        self.nominal_col = nominal_col
        self.non_nominal_col = non_nominal_col

    def fit(self, X: pd.DataFrame, y=None):

        col_tf = ColumnTransformer(
            [
                ('non_nominal', MinMaxScaler(), self.non_nominal_col),
                ('nominal', OneHotEncoder(), self.nominal_col)
            ]
        )

        col_tf.fit(X)
        self.col_tf = col_tf

        self.new_nominal_col = self._identify_new_column()

        return self

    def transform(self, X: pd.DataFrame):
        transformed_X = self.col_tf.transform(X)
        return pd.DataFrame(
            transformed_X, 
            columns=self.non_nominal_col + self.new_nominal_col)

    def _identify_new_column(self):
        # identify new column name
        new_nominal_col: t.List[str] = []

        for i in range(len(self.nominal_col)):
            temp = [f'{self.nominal_col[i]}_{sub}' for sub in self.col_tf.transformers_[1][1].categories_[i]]
            new_nominal_col += temp

        return new_nominal_col

    def inverse_transform(self, X: pd.DataFrame):

        non_nominal_array = self.col_tf.transformers_[0][1].inverse_transform(X[self.non_nominal_col])
        non_nominal_df = pd.DataFrame(non_nominal_array, columns=self.non_nominal_col)

        nominal_df = X[self.new_nominal_col]

        return pd.concat([non_nominal_df, nominal_df], axis=1)


class NominalEncoder(BaseEstimator, TransformerMixin):
    '''Implement one hot encoder to encode nominal features on the dataframe,
    passthrough all non-nominal features and preserving column names of the dataframe
    '''

    def __init__(self, *, nominal_col: t.List[str], non_nominal_col: t.List[str]):
        self.nominal_col = nominal_col
        self.non_nominal_col = non_nominal_col

        # these attributes will be generated after fitting
        self.new_nominal_col = None
        self.col_tf = None

        return None

    def fit(self, X: pd.DataFrame, y:t.Optional[np.ndarray]=None):

        # implement column transformer to just encode the nominal features
        # and passthrough all non nominal features
        col_tf = ColumnTransformer(
            [
                ('non_nominal', 'passthrough', self.non_nominal_col),
                ('nominal', OneHotEncoder(), self.nominal_col)
            ]
        )

        # fit and memorize the transformer and new columns name 
        col_tf.fit(X)
        self.col_tf = col_tf
        self.new_nominal_col = self._identify_new_column()

        return self

    def transform(self, X: pd.DataFrame):
        transformed_X = self.col_tf.transform(X)
        return pd.DataFrame(
            transformed_X, 
            columns=self.non_nominal_col + self.new_nominal_col)

    def _identify_new_column(self):
        # identify new column name
        new_nominal_col: t.List[str] = []

        for i in range(len(self.nominal_col)):
            # sub categories of each nominal features will be recored in
            # onehot encoder transformer which reside in col_tf.transfomers_
            # at the index [1][1]
            temp = [f'{self.nominal_col[i]}_{sub}' for sub in self.col_tf.transformers_[1][1].categories_[i]]
            new_nominal_col += temp

        return new_nominal_col


class EncodeUnknownTransformer(BaseEstimator, TransformerMixin):
    '''Transformer to encode unknown value to np.nan
    '''

    def __init__(self, feature2unk: t.Dict[str, t.Optional[t.List[float]]]):
        self.feature2unk = feature2unk

    def fit(self, X: t.Optional[pd.DataFrame]=None, y: t.Optional[np.ndarray]=None):
        return self

    def transform(self, X: pd.DataFrame):

        copy_X = X.copy()

        for feature in X.columns:
            copy_X[feature] = self._encode_unk(X, feature, self.feature2unk)

        return copy_X


    def _encode_unk(self, df: pd.DataFrame, feature: str, feature2unk: t.Dict[str, t.Optional[t.List[float]]]):

        col = df[feature].copy()

        if feature in feature2unk and feature2unk[feature] is not None:
            col.loc[col.isin(feature2unk[feature])] = np.nan

        return col

class SelectColumnsTransformer(BaseEstimator, TransformerMixin):
    '''Transformer for selecting sub set of column from pandas input
    and return output as pandas while preserve column name
    '''

    def __init__(self, selected_features: t.List[str]):
        self.selected_features = selected_features[:]
        return None

    def fit(self, X: t.Optional[pd.DataFrame]=None, y: t.Optional[np.ndarray]=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X[self.selected_features].copy()


class PandasImputer(BaseEstimator, TransformerMixin):
    '''Wrap median imputer so that the output still preserve column name
    '''

    def __init__(self, imputer: SimpleImputer):
        self.imputer = imputer
        return None

    def fit(self, X: pd.DataFrame, y: t.Optional[np.ndarray]=None):
        self.imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        return pd.DataFrame(self.imputer.transform(X), columns=X.columns)