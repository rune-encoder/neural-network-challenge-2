"""
Utilities package for data analysis and visualization.
"""

from .print_utils import print_title, print_label, print_footer
from .utils import highlight_vif, calc_vif, highlight_p_values, calc_p_values, calc_correlation
from .preprocessing_utils import encode_categorical_data, encode_categorical_target