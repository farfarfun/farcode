# @Time    : 2019/05/23 16:39
# @Author  : niuliangtao
# @Site    :
# @File    : __init__.py.py
# @Software: PyCharm

"""farcode 数据处理公开接口。"""

from .data import ElectronicsData, get_adult_data
from .download import download_file

__all__ = ["ElectronicsData", "download_file", "get_adult_data"]
