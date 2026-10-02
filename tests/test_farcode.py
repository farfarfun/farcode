import pickle
from pathlib import Path

import pytest

# farcode.attention_decoder 依赖未声明为默认依赖的旧版 tensorflow/keras
# （见 pyproject.toml 的 legacy-dl extra），默认环境下不保证可导入，故不在此测试。
import farcode
from farcode.data.data import ElectronicsData, get_adult_data
from farcode.data.download import download_file


def test_import_farcode():
    assert farcode is not None


def test_data_module_public_api_importable():
    assert callable(get_adult_data)
    assert callable(download_file)
    assert ElectronicsData is not None


def test_download_file_skips_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "existing"
    path.write_text("kept")
    called = False

    def fail_if_called(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("farcode.data.download.simple_download", fail_if_called)
    download_file("https://example.invalid/data", str(path))

    assert not called
    assert path.read_text() == "kept"


def test_download_file_propagates_download_failure(tmp_path, monkeypatch):
    def fail(**_kwargs):
        raise OSError("network failed")

    monkeypatch.setattr("farcode.data.download.simple_download", fail)
    with pytest.raises(OSError, match="network failed"):
        download_file("https://example.invalid/data", str(tmp_path / "missing"))


def test_data_root_argument_precedes_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("FARCODE_DATA_ROOT", str(tmp_path / "environment"))

    data = ElectronicsData(tmp_path / "argument")

    assert data.path_root == tmp_path / "argument" / "data" / "electronics"


def test_get_adult_data_normal_path_without_network(tmp_path, monkeypatch):
    train_rows = [
        "39, Private, 1, Bachelors, 13, Never-married, Tech, Own-child, White, Male, 0, 0, 40, US, <=50K",
        "50, Private, 2, Bachelors, 13, Married, Tech, Husband, White, Male, 10, 0, 50, US, >50K",
    ]
    test_rows = [
        "40, Private, 3, Bachelors, 13, Never-married, Tech, Own-child, White, Male, 0, 0, 40, US, <=50K.",
        "51, Private, 4, Bachelors, 13, Married, Tech, Husband, White, Male, 10, 0, 50, US, >50K.",
    ]

    def write_fixture(*, url, path):
        rows = test_rows if "test" in url else train_rows
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(rows) + "\n")

    monkeypatch.setattr("farcode.data.data.download_file", write_fixture)
    train_x, train_y, test_x, test_y = get_adult_data(tmp_path)

    assert train_x.shape[0] == test_x.shape[0] == 2
    assert train_x.columns.tolist() == test_x.columns.tolist()
    assert train_y.tolist() == [0, 1]
    assert test_y.tolist() == [0, 1]


def test_convert_pd_parses_json_and_legacy_literal(tmp_path):
    data = ElectronicsData(tmp_path)
    data.path_root.mkdir(parents=True)
    (data.path_root / "reviews_Electronics_5.json").write_text(
        '{"reviewerID": "u1", "asin": "a1", "unixReviewTime": 1}\n'
    )
    (data.path_root / "meta_Electronics.json").write_text(
        "{'asin': 'a1', 'categories': [['Electronics']]}\n"
    )

    data.convert_pd_1()

    with (data.path_root / "raw_data/reviews.pkl").open("rb") as file:
        reviews = pickle.load(file)
    with (data.path_root / "raw_data/meta.pkl").open("rb") as file:
        metadata = pickle.load(file)
    assert reviews.loc[0, "reviewerID"] == "u1"
    assert metadata.loc[0, "categories"] == [["Electronics"]]


@pytest.mark.parametrize(
    ("content", "message", "has_line_number"),
    [
        ("", "不包含任何记录", False),
        ("not valid\n", "无法解析", True),
        ('{"asin": "a1"}\n', "缺少字段: categories", True),
    ],
)
def test_convert_pd_reports_file_and_line_for_bad_metadata(
    tmp_path, content, message, has_line_number
):
    data = ElectronicsData(tmp_path)
    data.path_root.mkdir(parents=True)
    (data.path_root / "reviews_Electronics_5.json").write_text(
        '{"reviewerID": "u1", "asin": "a1", "unixReviewTime": 1}\n'
    )
    metadata_path = data.path_root / "meta_Electronics.json"
    metadata_path.write_text(content)

    with pytest.raises(ValueError, match=message) as exc_info:
        data.convert_pd_1()

    assert str(metadata_path) in str(exc_info.value)
    assert ("第 1 行" in str(exc_info.value)) is has_line_number
