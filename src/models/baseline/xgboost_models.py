# src/models/xgboost_model.py
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from ..base_model import BaseForecastingModel
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

class XGBoostForecastingModel(BaseForecastingModel):
    """
    XGBoost regression model for demand forecasting.
    
    Teaching points:
    - XGBoost handles non-linearities and interactions automatically.
    - Sparse or high-dimensional data can be handled efficiently.
    - Preprocessing (imputation, scaling) ensures robust predictions.
    - Predictions are clamped to non-negative values for demand forecasting.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("XGBoost", config)
        self.config = config or {}
        self.imputer = None
        self.scaler = None
        
        # XGBoost hyperparameters with sensible defaults
        self.params = {
            'n_estimators': self.config.get('n_estimators', 200),
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.use_scaling = self.config.get('use_scaling', True)
        self.scaling_method = self.config.get('scaling_method', 'standard')
        
        self.model = xgb.XGBRegressor(**self.params)
    
    def _prepare_features(self, X: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        """
        Handle missing values, scale features, and ensure numeric only.
        """
        X_processed = X.copy()
        
        # Only numeric columns
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed = X_processed[numeric_cols]
        self.feature_names = numeric_cols.tolist()
        
        # Handle missing values
        if is_training:
            self.imputer = SimpleImputer(strategy='median')
            X_processed = pd.DataFrame(
                self.imputer.fit_transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        else:
            if self.imputer is None:
                raise ValueError("Imputer not fitted yet")
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
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'XGBoostForecastingModel':
        self.validate_input(X, y)
        
        # Prepare features
        X_processed = self._prepare_features(X, is_training=True)
        
        # Handle extreme values
        X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Fit XGBoost
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        # Store training metrics
        train_score = self.model.score(X_processed, y)
        self.training_history['train_r2'] = train_score
        self.training_history['n_features'] = X_processed.shape[1]
        self.training_history['n_samples'] = X_processed.shape[0]
        
        logger.info(f"XGBoost training completed. R² score: {train_score:.4f}")
        return self
    
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_processed = self._prepare_features(X, is_training=False)
        X_processed = np.nan_to_num(X_processed, nan=0.0, posinf=1e10, neginf=-1e10)
        
        preds = self.model.predict(X_processed)
        # Ensure non-negative predictions
        preds = np.maximum(preds, 0)
        return preds
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Extract XGBoost feature importances.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}
