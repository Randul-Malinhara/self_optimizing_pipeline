# src/concept_drift.py
import numpy as np
import pandas as pd
from river.drift import ADWIN

class DriftDetector:
    """
    A simple concept drift detector using ADWIN.
    """

    def __init__(self, delta=0.002):
        self.adwin = ADWIN(delta=delta)

    def update(self, y_true: float, y_pred: float) -> bool:
        """
        Update drift detector with new data point and check if drift occurred.

        Args:
            y_true (float): Actual label
            y_pred (float): Predicted label

        Returns:
            bool: True if drift is detected, False otherwise.
        """
        error = int(y_true != y_pred)
        in_drift, _ = self.adwin.update(error)
        return in_drift
