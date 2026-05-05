import pandas as pd
from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    """
    Abstract base class for all FrustraMotion analytical engines.
    Handles common data loading, validation, and shared utilities.
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self._validate_columns()
        self.total_frames = self.df['Frame'].nunique()

    @abstractmethod
    def _validate_columns(self):
        """
        Ensures the DataFrame has the correct base format.
        Child classes MUST override this method to add specific checks.
        """
        if 'Frame' not in self.df.columns or 'FrstIndex' not in self.df.columns:
            raise ValueError("[!] Error: DataFrame must contain 'Frame' and 'FrstIndex' columns.")

    def _get_noise_threshold(self, min_frequency=0.1):
        """
        Calculates the minimum number of frames required to consider an entity valid.
        Used to filter out highly transient noise.
        """
        return self.total_frames * min_frequency