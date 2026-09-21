# @Time    : 2019/05/23 16:53
# @Author  : niuliangtao
# @Site    :
# @File    : download.py
# @Software: PyCharm
import os

from farlog import getLogger
from funget import simple_download

__all__ = ["download_file"]

logger = getLogger(__name__)


def download_file(url: str, path: str) -> None:
    """下载文件到本地路径（已存在则跳过）。

    Args:
        url: 下载链接。
        path: 保存路径。
    """
    if not os.path.exists(path):
        logger.info(f"downloading from {url} to {path}")
        simple_download(url=url, filepath=path)
        logger.info("download success")
