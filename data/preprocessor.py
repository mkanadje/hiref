import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import pickle
import config
from data.constraint_helper import get_constraint_indices


class DataPreprocessor:
    def __init__(self, hierarchy_cols, feature_cols, target_col, key_col, date_col):
        self.hierarchy_cols = hierarchy_cols
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.key_col = key_col
        self.date_col = date_col

        # Each hierarchy to be encoded
        self.label_encoders = {}
        self.feature_scaler = config.FEATURE_SCALER
        self.target_scaler = config.TARGET_SCALER
        self.vocab_sizes = {}
        self.constraint_indices = None

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
        # self.feature_scaler = StandardScaler()
        # self.feature_scaler = MinMaxScaler()
        self.feature_scaler.fit(df[self.feature_cols])

        # Fit StandarScaler to target
        # self.target_scaler = StandardScaler()
        # self.target_scaler = MinMaxScaler()
        self.target_scaler.fit(df[[self.target_col]])

        # Compute constraint indices if constraints are enabled
        if config.USE_CONSTRAINT_WEIGHTS:
            self.constraint_indices = get_constraint_indices(
                self.feature_cols,
                config.POSITIVE_WEIGHT_FEATURES,
                config.NEGATIVE_WEIGHT_FEATURES,
            )
            print(f"\n {'='*60}")
            print("WEIGHT CONSTRAINT CONFIGURATION")
            print(f"{'='*60}")
            print(
                f"Positive constraints({len(self.constraint_indices["positive_indices"])} features): "
            )
            for idx in self.constraint_indices["positive_indices"]:
                print(f" [{idx} {self.feature_cols[idx]}]")
            print(
                f"Negative constraints({len(self.constraint_indices["negative_indices"])} features): "
            )
            for idx in self.constraint_indices["negative_indices"]:
                print(f" [{idx} {self.feature_cols[idx]}]")
            print(
                f"Unconstrained constraints({len(self.constraint_indices["unconstrained_indices"])} features): "
            )
            for idx in self.constraint_indices["unconstrained_indices"]:
                print(f" [{idx} {self.feature_cols[idx]}]")
        else:
            print("\n Weight constraint DISABLED - using standard unconstrained model")
        return self

    def transform(self, df):
        hierarchy_ids = {
            c: self.label_encoders[c].transform(df[c]) for c in self.hierarchy_cols
        }
        features = self.feature_scaler.transform(df[self.feature_cols])
        targets = self.target_scaler.transform(df[[self.target_col]])
        if isinstance(targets, pd.DataFrame):
            targets = targets.values
        targets = targets.flatten()
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
                    "constraint_indices": self.constraint_indices,
                    "feature_cols": self.feature_cols,
                    "hierarchy_cols": self.hierarchy_cols,
                    "target_col": self.target_col,
                    "key_col": self.key_col,
                    "date_col": self.date_col,
                },
                f,
            )

    @classmethod
    def load(cls, file_path):
        """
        Load a saved preprocessor from disk.

        Can be called as:
        - Class method: preprocessor = DataPreprocessor.load(path)
        - Instance method: preprocessor.load(path) returns self
        """
        # Load pickled state
        with open(file_path, "rb") as f:
            state = pickle.load(f)

        # If called on an instance (self is not None in that context),
        # we modify the instance. Otherwise create new instance.
        # However, classmethods don't have access to self, so we always create new
        instance = cls.__new__(cls)

        # Manually set the attributes
        instance.label_encoders = state["label_encoders"]
        instance.feature_scaler = state["feature_scaler"]
        instance.target_scaler = state["target_scaler"]
        instance.vocab_sizes = state["vocab_sizes"]
        # Load constraint_indices (with backward compatibility for old preprocessors)
        instance.constraint_indices = state.get("constraint_indices", None)
        # Load column configurations (with backward compatibility)
        instance.feature_cols = state.get("feature_cols", None)
        instance.hierarchy_cols = state.get(
            "hierarchy_cols", list(state["label_encoders"].keys())
        )
        instance.target_col = state.get("target_col", None)
        instance.key_col = state.get("key_col", None)
        instance.date_col = state.get("date_col", None)
        return instance

    def get_vocab_sizes(self):
        return self.vocab_sizes
