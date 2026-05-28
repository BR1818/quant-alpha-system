"""选股模块"""

from modules.selectors.composite_selector import CompositeSelector
from modules.selectors.xgboost_selector import XGBoostSelector

__all__ = ["CompositeSelector", "XGBoostSelector"]
