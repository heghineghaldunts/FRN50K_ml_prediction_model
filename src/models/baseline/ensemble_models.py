# src/models/ensemble_model.py
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from ..base_model import BaseForecastingModel
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

class EnsembleForecastingModel(BaseForecastingModel):
    """
    Ensemble regression model combining multiple regressors for robust demand forecasting.

    Teaching points:
    - Ensembles reduce variance and improve generalization.
    - Preprocessing (imputation + scaling) ensures robustness for sparse/high-dimensional data.
    - Predictions are clamped to non-negative values.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("EnsembleRegression", config)
        self.config = config or {}
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        
        # Ensemble components
        rf_params = self.config.get('rf_params', {})
        gb_params = self.config.get('gb_params', {})
        xgb_params = self.config.get('xgb_params', {})
        
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42, **rf_params)
        gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42, **gb_params)
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                                     colsample_bytree=0.8, random_state=42, n_jobs=-1, **xgb_params)
        
        self.model = VotingRegressor([
            ('rf', rf_model),
            ('gb', gb_model),
            ('xgb', xgb_model)
        ])
        
        self.use_scaling = self.config.get('use_scaling', True)
        self.scaling_method = self.config.get('scaling_method', 'standard')
    
    def _prepare_features(self, X: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        """
        Handle missing values, scale features, and keep numeric columns only.
        """
        X_processed = X.copy()
        
        # Keep numeric columns only
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed = X_processed[numeric_cols]
        
        # Imputation
        if is_training:
            self.all_nan_cols = X_processed.columns[X_processed.isna().all()]
            if len(self.all_nan_cols) > 0:
                X_processed = X_processed.drop(columns=self.all_nan_cols)

            self.imputer = SimpleImputer(strategy='median')
            X_processed = pd.DataFrame(
                self.imputer.fit_transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )

            self.feature_names_ = X_processed.columns.tolist()

        else:
            if self.imputer is None:
                raise ValueError("Imputer not fitted yet")
            if hasattr(self, "all_nan_cols"):
                X_processed = X_processed.drop(columns=self.all_nan_cols, errors="ignore")

            X_processed = X_processed[self.feature_names_]
            X_processed = pd.DataFrame(
                self.imputer.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        
        # Scaling
        if self.use_scaling:
            if is_training:
                self.scaler = StandardScaler() if self.scaling_method=='standard' else RobustScaler()
                X_scaled = self.scaler.fit_transform(X_processed)
            else:
                if self.scaler is None:
                    raise ValueError("Scaler not fitted yet")
                X_scaled = self.scaler.transform(X_processed)
        else:
            X_scaled = X_processed.values
        
        return X_scaled
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'EnsembleForecastingModel':
        self.validate_input(X, y)
        
        X_processed = self._prepare_features(X, is_training=True)
        X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Fit ensemble
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        # Store training metrics
        train_score = self.model.score(X_processed, y)
        self.training_history['train_r2'] = train_score
        self.training_history['n_features'] = X_processed.shape[1]
        self.training_history['n_samples'] = X_processed.shape[0]
        
        logger.info(f"Ensemble regression training completed. R² score: {train_score:.4f}")
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_processed = self._prepare_features(X, is_training=False)
        X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1e10, neginf=-1e10)
        
        preds = self.model.predict(X_processed)
        # Clamp predictions to non-negative
        preds = np.maximum(preds, 0)
        return preds
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Aggregate feature importances from ensemble models (average over components if available).
        """
        if not hasattr(self.model, "estimators_"):
            return None

        importances = {}

        for estimator in self.model.estimators_:
            if hasattr(estimator, "feature_importances_"):
                for i, f in enumerate(self.feature_names_):
                    importances[f] = importances.get(f, 0.0) + estimator.feature_importances_[i]

        # Average
        for f in importances:
            importances[f] /= len(self.model.estimators_)

        return importances
