"""Marketing Intelligence Skills.

Each skill takes a snapshot dict and returns a list of finding dicts.
"""

from .acquisition import analyze as analyze_acquisition
from .conversion import analyze as analyze_conversion
from .contribution import analyze as analyze_contribution

__all__ = ["analyze_acquisition", "analyze_conversion", "analyze_contribution"]
