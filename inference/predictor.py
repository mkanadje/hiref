from data.preprocessor import DataPreprocessor
from models.hierarchical_model import HierarchicalModel
import config
import torch
from tqdm import tqdm


class Predictor:
    def __init__(self, model_path, preprocessor_path, device="cpu"):
        """
        Load the already trained model and preprocessors for predictions
        """
        # Device safety check
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            if device != "cpu":
                print(f"Warning: {device} is not available, using CPU")
        self.preprocessor = DataPreprocessor.load(preprocessor_path)

        # We have to create the exact model architecture where we will use the trained model.
        vocab_sizes = self.preprocessor.get_vocab_sizes()
        embedding_dims = config.EMBEDDING_DIMS
        n_features = len(config.FEATURE_COLS)
        print(f"Creating model architecture...")
        print(f"Vocab size: {vocab_sizes}")
        print(f"Embedding dims: {embedding_dims}")
        print(f"N features: {n_features}")
        # Create model with same architecture
        self.model = HierarchicalModel(
            vocab_sizes=vocab_sizes,
            embedding_dims=embedding_dims,
            n_features=n_features,
        )
        # Load trained weights
        print(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
        self.feature_cols = config.FEATURE_COLS
        self.hierarchy_cols = config.HIERARCHY_COLS

    def predict_single(self, sku_dict):
        """
        Predict sales for a single SKU

        Args:
            sku_dict: Dictionary containing hierarchy and features
                Example: {
                        "region": "Northeast",
                        "state": "Massachusetts",
                        "segement": "Premium",
                        "temperature": 72.5,
                        "price": 4.99,
                        # .... other features
                    }
        Returns:
            dict: {
                "prediction": 156.7, # Prediction sales in original scale
                "baseline": 85.0, # Bias + embedding contributions
                "bias": 10.0, # Bias term alone
                "embedding_contribution": 75.0, # Total from all embeddings
                "driver_contributions": {
                    "temperature": 12.5,
                    "precipitation": -3.2,
                    "price": -8.7,
                    "tv_spend": 25.3,
                    # ... for each feature
                },
                "total_driver_impact": 71.7, # Sum of all driver contributions
                "sku_key": "Northeast_Massachusetts_Premium_Alpine Springs_8oz",
            }
        """
        import numpy as np
        import pandas as pd

        # DATE column must be present
        if "date" not in sku_dict:
            raise ValueError(
                "'date' is a required field in the input data for forecasting."
            )
        date_value = sku_dict["date"]

        # Step 1: Preprocess the input
        # 1a. Extract hierarchy values and encode
        hierarchy_ids = {}
        for col in self.hierarchy_cols:
            value = sku_dict[col]
            encode_id = self.preprocessor.label_encoders[col].transform([value])[0]
            hierarchy_ids[col] = torch.tensor([encode_id], dtype=torch.long)
        # 1b. Extract and scale features
        feature_values = [sku_dict[col] for col in self.feature_cols]
        feature_array = np.array(feature_values).reshape(1, -1)  # (1, n_features)
        feature_df = pd.DataFrame([feature_values], columns=self.feature_cols)
        scaled_features = self.preprocessor.feature_scaler.transform(feature_df)
        features = torch.tensor(scaled_features, dtype=torch.float32)

        # Step 2: Move to device
        for col in self.hierarchy_cols:
            hierarchy_ids[col] = hierarchy_ids[col].to(self.device)
        features = features.to(self.device)

        # Step 3: Get prediction and contributions
        with torch.no_grad():
            predictions_scaled = self.model(hierarchy_ids, features)
            contributions = self.model.get_prediction_contributions(
                hierarchy_ids, features
            )

        # Step 4: Inverse transform prediction to original scale
        predictions_scaled_np = predictions_scaled.cpu().numpy().reshape(1, -1)
        prediction_original = self.preprocessor.target_scaler.inverse_transform(
            predictions_scaled_np
        ).flatten()[0]

        # Step 5: Extract contribution in the scaled space
        bias = contributions["bias_contribution"]
        embedding_contribution_scaled = (
            contributions["embedding_contribution"].cpu().item()
        )
        baseline_scaled = bias + embedding_contribution_scaled

        # Extract individual driver contribution scaled
        feature_breakdown = contributions["feature_breakdown"]
        driver_contribution_scaled = {}
        for i, feature_name in enumerate(self.feature_cols):
            contrib_value = feature_breakdown[f"feature_{i}"].cpu().item()
            driver_contribution_scaled[feature_name] = contrib_value
        total_driver_impact_scaled = sum(driver_contribution_scaled.values())

        # Step 6: Convert driver contribution to original scale
        # feature_stds = (
        #     self.preprocessor.feature_scaler.scale_
        # )  # (n_features, ) Standatd Dev of features
        target_std = self.preprocessor.target_scaler.scale_[0]
        driver_contribution_original = {}
        for i, feature_name in enumerate(self.feature_cols):
            contrib_scaled = driver_contribution_scaled[feature_name]
            # contrib_original = contrib_scaled * feature_stds[i]
            contrib_original = contrib_scaled * target_std
            driver_contribution_original[feature_name] = float(contrib_original)
        total_driver_impact_original = sum(driver_contribution_original.values())

        # Calculate baseline as a residual of total driver impact to maintain additivity
        baseline_original = prediction_original - total_driver_impact_original

        sku_key = "_".join([str(sku_dict[col]) for col in self.hierarchy_cols])

        # Step 7: Build result
        result = {
            # Temporal information
            "date": date_value,
            # Prediction in original space
            "prediction": float(prediction_original),
            # Baseline component in  scaled space
            "baseline_scaled": float(baseline_scaled),
            # Baseline component in original space (for business use) calculated as residual
            "baseline_original": float(baseline_original),
            "bias": float(bias),
            "embedding_contribution_scaled": float(embedding_contribution_scaled),
            # Driver contribution in SCALED space (for model debugging)
            "driver_contribution_scaled": driver_contribution_scaled,
            "total_driver_contribution_scaled": float(total_driver_impact_scaled),
            # Driver contribution in ORIGINAL space (for business interpretation)
            "driver_contribution_original": driver_contribution_original,
            "total_driver_contribution_original": float(total_driver_impact_original),
            # Metadata
            "sku_key": sku_key,
        }
        return result

    def predict_batch(self, sku_data, return_dataframe=False, show_progress=False):
        """
        Predict for multiple SKUs using vectorized batch processing

        All SKUs are processed in a single forward pass for maximum efficiency.
        Supports both list of dictionaries and pandas DataFrame as input

        Args:
            sku_data: Either:
                - List of SKU dictionaries, each containing 'date', hierarchy, and feature values
                - pandas DataFrame with 'date', hierarchy, and feature column
            return_dataframe: If True, return pandas DataFrame, else list of dicts
            show_progress: If True, display progress bar during result unpacking

        Raise:
            ValueError: If 'date' column/field is missing from input data

        Example:
            >>> sku_list = [
            ....    {"date":"2024-12-01", "region": "Northeast", "state": "Northfield", ....},
            ....    {"date":"2024-12-01", "region": "Southeast", "state": "Suncoast", ....},
            ....]
            >>> results = predictor.predict_batch(sku_list)
            >>> # Or as a dataframe
            >>> results_df = predictor.predict_batch(sku_df, return_dataframe=True)
            >>> # With progress tracking for large batches
            >>> results_df = predictor.predict_batch(sku_df, return_dataframe=True, show_progress=True)
        """
        import pandas as pd
        import numpy as np

        # Step 1: Normalize input to DataFrame
        if isinstance(sku_data, pd.DataFrame):
            sku_df = sku_data
            input_was_df = True
        else:
            # Convert list of dicts to DataFrame for columnar processing
            sku_df = pd.DataFrame(sku_data)
            input_was_df = False

        # Validate: DATE column must be present
        if "date" not in sku_df.columns:
            raise ValueError(
                "'date' column is required in input data for forecasting. Forecasts mush specify the timeperiod they are for"
            )
        batch_size = len(sku_df)
        if batch_size == 0:
            return pd.DataFrame() if return_dataframe else []

        # Step 2: Batch encode all hierarchical columns
        hierarchy_ids = {}
        for col in self.hierarchy_cols:
            values = sku_df[col].values
            encoded_ids = self.preprocessor.label_encoders[col].transform(values)
            hierarchy_ids[col] = torch.tensor(encoded_ids, dtype=torch.long)
        # Step 3: Batch extract and scale features
        feature_df = sku_df[self.feature_cols]
        scaled_features = self.preprocessor.feature_scaler.transform(feature_df)
        features = torch.tensor(scaled_features, dtype=torch.float32)
        # Step 4: Move all tensors to device at once
        for col in self.hierarchy_cols:
            hierarchy_ids[col] = hierarchy_ids[col].to(self.device)
        features = features.to(self.device)
        # Step 5: Single forward pass for entire batch
        with torch.no_grad():
            prediction_scaled = self.model(hierarchy_ids, features)
            contributions = self.model.get_prediction_contributions(
                hierarchy_ids, features
            )
        # Step 6: Inverse transofrm all predictions at once
        prediction_scaled_np = (
            prediction_scaled.cpu().numpy().reshape(-1, 1)
        )  # (batch_size, 1)
        predictions_original = self.preprocessor.target_scaler.inverse_transform(
            prediction_scaled_np
        ).flatten()

        # Step 7: Extract batched contribution
        bias = contributions["bias_contribution"]
        embedding_contributions_scaled = (
            contributions["embedding_contribution"].cpu().numpy()
        )
        feature_breakdown_scaled = contributions["feature_breakdown"]
        # feature_stds = self.preprocessor.feature_scaler.scale_
        target_std = self.preprocessor.target_scaler.scale_[0]

        # Step 8: VECTORIZED unpacking
        # Convert all driver contributions to DataFrame at once
        driver_contrib_df_scaled = pd.DataFrame(
            {
                f"driver_{feat}_scaled": feature_breakdown_scaled[f"feature_{i}"]
                .cpu()
                .numpy()
                for i, feat in enumerate(self.feature_cols)
            }
        )
        driver_contrib_df_original = pd.DataFrame(
            {
                f"driver_{feat}_original": feature_breakdown_scaled[f"feature_{i}"]
                .cpu()
                .numpy()
                # * feature_stds[i]
                * target_std
                for i, feat in enumerate(self.feature_cols)
            }
        )
        # Calculate totals vectorized
        total_driver_scaled = driver_contrib_df_scaled.sum(axis=1).values
        total_driver_original = driver_contrib_df_original.sum(axis=1).values

        sku_key = sku_df[self.hierarchy_cols[0]].astype(str)
        for col in self.hierarchy_cols[1:]:
            sku_key = sku_key + "_" + sku_df[col].astype(str)
        # Build result DataFRame directly
        results_df = pd.DataFrame(
            {
                "date": sku_df["date"].values,
                "sku_key": sku_key.values,
                "prediction": predictions_original,
                "baseline_scaled": bias + embedding_contributions_scaled,
                "baseline_original": predictions_original - total_driver_original,
                "bias": bias,
                "embedding_contribution_scaled": embedding_contributions_scaled,
                "total_driver_contribution_scaled": total_driver_scaled,
                "total_driver_contribution_original": total_driver_original,
            }
        )

        # Add driver columns
        results_df = pd.concat(
            [results_df, driver_contrib_df_scaled, driver_contrib_df_original], axis=1
        )
        if return_dataframe:
            return results_df
        else:
            return results_df.to_dict("records")
