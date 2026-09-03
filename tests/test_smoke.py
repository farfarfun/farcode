# funcode.attention_decoder 依赖未声明为默认依赖的旧版 tensorflow/keras
# （见 pyproject.toml 的 legacy-dl extra），默认环境下不保证可导入，故不在此测试。
import funcode
from funcode.data.data import ElectronicsData, get_adult_data
from funcode.data.download import download_file


def test_import_funcode():
    assert funcode is not None


def test_data_module_public_api_importable():
    assert callable(get_adult_data)
    assert callable(download_file)
    assert ElectronicsData is not None
