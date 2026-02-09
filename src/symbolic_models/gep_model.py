import numpy as np

from .base_model import BaseRegressionModel


class GEPModel(BaseRegressionModel):
    """
    Placeholder for GEP regression model, not yet implemented
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the data"""
        return NotImplementedError("GEPModel.fit() not implemented")

    def predict(self, X: np.ndarray):
        """Run inference on the model"""
        return NotImplementedError("GEPModel.predict() not implemented")

    def get_expression(self):
        """Return string of math expression from the model"""
        return NotImplementedError("GEPModel.get_expression() not implemented")
