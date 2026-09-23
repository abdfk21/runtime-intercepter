"""Inter-Annotator Agreement statistics including Fleiss' and Cohen's Kappa."""

from typing import Dict, List, Optional
import numpy as np


class AgreementCalculator:
    """Calculates inter-rater reliability metrics."""

    @staticmethod
    def calculate_fleiss_kappa(rating_matrix: np.ndarray) -> float:
        """Computes Fleiss' Kappa for multiple raters.
        
        Args:
            rating_matrix: N x k array where N is number of subjects,
                           k is number of categories, and entries are count of raters.
        """
        N, k = rating_matrix.shape
        n_raters = np.sum(rating_matrix[0, :])
        if n_raters <= 1:
            return 0.0

        p_j = np.sum(rating_matrix, axis=0) / (N * n_raters)
        P_e = np.sum(p_j ** 2)

        P_i = (np.sum(rating_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
        P_bar = np.mean(P_i)

        if 1 - P_e == 0:
            return 1.0

        kappa = (P_bar - P_e) / (1 - P_e)
        return float(round(kappa, 4))
