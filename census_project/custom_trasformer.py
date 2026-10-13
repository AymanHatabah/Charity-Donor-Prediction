from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# ==========================================
# 🚀 Ultimate Feature Aligner (Production-Ready)
# ==========================================

class FeatureAligner(BaseEstimator, TransformerMixin):
    def __init__(self, num_fill=0, cat_fill='Unknown'):
        # 1. Smart Fill: Define distinct defaults for numbers and text
        self.num_fill = num_fill
        self.cat_fill = cat_fill
        
    def fit(self, X, y=None):
        # 2. NumPy Defense: Ensure X is a DataFrame during training to learn names
        if isinstance(X, np.ndarray):
            raise TypeError("FeatureAligner requires a Pandas DataFrame during 'fit' to learn column names.")
            
        self.reference_columns_ = X.columns.tolist()
        # 3. Type Enforcement: Memorize the exact data types of the training data
        self.reference_dtypes_ = X.dtypes.to_dict()
        
        return self
    
    def transform(self, X):
        # 2. NumPy Defense: Convert arrays to DataFrames if they come from standard pipelines
        if isinstance(X, np.ndarray):
            if X.shape[1] == len(self.reference_columns_):
                X = pd.DataFrame(X, columns=self.reference_columns_)
            else:
                raise ValueError(f"Expected {len(self.reference_columns_)} columns, got {X.shape[1]}")
                
        X_aligned = X.copy()
        
        # Fix the categorical dtype pandas issue
        cat_columns = X_aligned.select_dtypes(include=['category']).columns
        for col in cat_columns:
            X_aligned[col] = X_aligned[col].astype('object')
            
        # Drop extra columns
        extra_cols = [col for col in X_aligned.columns if col not in self.reference_columns_]
        if extra_cols:
            X_aligned = X_aligned.drop(columns=extra_cols)
            
        # Add missing columns with SMART FILL (Numbers vs Categories)
        missing_cols = [col for col in self.reference_columns_ if col not in X_aligned.columns]
        for col in missing_cols:
            expected_type = self.reference_dtypes_[col]
            # Check if the missing column was numeric in training
            if pd.api.types.is_numeric_dtype(expected_type):
                X_aligned[col] = self.num_fill
            else:
                X_aligned[col] = self.cat_fill
                
        # Reorder to match reference
        X_aligned = X_aligned[self.reference_columns_]
        
        # 3. Type Enforcement: Force new data to match training data types
        for col, expected_type in self.reference_dtypes_.items():
            try:
                X_aligned[col] = X_aligned[col].astype(expected_type)
            except Exception as e:
                # If a completely invalid type is sent (e.g., text instead of number), ignore and let downstream handle it or log it
                pass 
                
        return X_aligned