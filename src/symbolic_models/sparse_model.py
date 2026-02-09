import numpy as np
import sklearn.linear_model as lm

from .base_model import BaseRegressionModel


class SparseRegressionModel(BaseRegressionModel):
    """
    Sparse regression model using a candidate set of features.
    """

    def __init__(self, model_type: str = "lasso", **kwargs) -> None:
        self.is_fitted = False
        self.model = None
        if model_type == "lasso":
            self.model = lm.Lasso(**kwargs)
        elif model_type == "ridge":
            self.model = lm.Ridge(**kwargs)
        elif model_type == "elasticnet":
            self.model = lm.ElasticNet(**kwargs)
        else:
            raise ValueError(f"Unknown model model_type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SparseRegressionModel":
        """
        Fit the sparse regression model to the data.

        Parameters:
            X (np.ndarray): The input data.
            y (np.ndarray): The target data.

        Returns:
            SparseRegressionModel: The fitted model.
        """
        self.is_fitted = True
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the input data.

        Parameters:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted target values.
        """
        assert self.is_fitted, "Model is not fitted"
        y_hat = self.model.predict(X)
        return y_hat

    def get_expression(self) -> str:
        return super().get_expression()
