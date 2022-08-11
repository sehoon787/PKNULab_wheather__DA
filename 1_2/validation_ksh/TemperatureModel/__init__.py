# __init__.py
# file maker -> ipr (이상치 제거) -> vif ols -> MLR
from .file_maker import FileMaker
from .testdata_maker import TestDataMaker
from .iqr import IQR
from .iqr import difference
from .iqr import outlier_iqr
from .IndependentVal import get_corrleation
from .IndependentVal import get_vif
from .IndependentVal import vif_ols