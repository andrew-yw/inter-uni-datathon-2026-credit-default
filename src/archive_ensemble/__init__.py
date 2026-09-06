"""Clean reconstruction of the archive's model-only ensemble.

This package contains no source-dataset matching or external labels.  It is
limited to feature engineering, estimators, and probability blending trained
from the organizer-provided training split.
"""

from .models import Recipe, fit_model

__all__ = ["Recipe", "fit_model"]
