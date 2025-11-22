import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle


class DataPreprocessor:
    def __init__(self, hierarchy_cols, feature_cols, target_col, key_col, date_col):
        self.hierarchy_cols = hierarchy_cols
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.key_col = key_col
        self.date_col = date_col

        # Each hierarchy to be encoded
        self.label_encoders = {}
        self.feature_scaler = (
            None  # TODO: Make dictionary so each feature can have a separate scaler
        )
        self.target_scaler = None
        self.vocab_sizes = {}

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        return df

    def create_key_column(self, df):
        key = df[self.hierarchy_cols[0]].astype(str)
        for col in self.hierarchy_cols[1:]:
            key = key + "_" + df[col].astype(str)
        df[self.key_col] = key
        return df

    def fit(self, df):
        for col in self.hierarchy_cols:
            le = LabelEncoder()
            le.fit(df[col])
            self.label_encoders[col] = le
            self.vocab_sizes[col] = len(le.classes_)

        # Fit StandardScaler for features
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(df[self.feature_cols])

        # Fit StandarScaler to target
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(df[[self.target_col]])
        return self

    def transform(self, df):
        hierarchy_ids = {
            c: self.label_encoders[c].transform(df[c]) for c in self.hierarchy_cols
        }
        features = self.feature_scaler.transform(df[self.feature_cols])
        targets = self.target_scaler.transform(df[[self.target_col]]).flatten()
        keys = df[self.key_col].values
        return hierarchy_ids, features, targets, keys

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def split_by_date(self, df, train_end_date, val_end_date):
        train_df = df[df[self.date_col] <= train_end_date].reset_index(drop=True)
        val_df = df[
            (df[self.date_col] > train_end_date) & (df[self.date_col] <= val_end_date)
        ].reset_index(drop=True)
        test_df = df[df[self.date_col] > val_end_date].reset_index(drop=True)
        return train_df, val_df, test_df

    def prepare_data(self, file_path, train_end_date, val_end_date):
        df = self.load_data(file_path)
        df = self.create_key_column(df)
        train_df, val_df, test_df = self.split_by_date(df, train_end_date, val_end_date)
        train_data = self.fit_transform(train_df)
        val_data = self.transform(val_df)
        test_data = self.transform(test_df)
        return train_data, val_data, test_data

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "label_encoders": self.label_encoders,
                    "feature_scaler": self.feature_scaler,
                    "target_scaler": self.target_scaler,
                    "vocab_sizes": self.vocab_sizes,
                },
                f,
            )

    @classmethod
    def load(cls, file_path):
        # Create an empty instance WITHOUT calling __init__()
        instance = cls.__new__(cls)
        # Load pickled state
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        # Manually set the attributes (bupassing the __init__() call)
        instance.label_encoders = state["label_encoders"]
        instance.feature_scaler = state["feature_scaler"]
        instance.target_scaler = state["target_scaler"]
        instance.vocab_sizes = state["vocab_sizes"]
        # Set other attributes that aren't saved but might be needed
        instance.hierarchy_cols = list(state["label_encoders"].keys())
        instance.feature_cols = None
        instance.target_col = None
        instance.key_col = None
        instance.date_col = None
        return instance

    def get_vocab_sizes(self):
        return self.vocab_sizes
