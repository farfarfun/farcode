# @Time    : 2019/03/30 12:10
# @Author  : niuliangtao
# @Site    :
# @File    : data.py
# @Software: PyCharm
import ast
import gzip
import json
import os
import random
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from farlog import getLogger
from sklearn.preprocessing import StandardScaler

from .download import download_file

logger = getLogger(__name__)

DEFAULT_DATA_ROOT = Path("/content/tmp")


def _data_root(path: str | Path | None = None) -> Path:
    """按参数、环境变量、TOML 配置和默认值顺序解析数据根目录。"""
    if path:
        return Path(path).expanduser()
    if value := os.getenv("FARCODE_DATA_ROOT"):
        return Path(value).expanduser()
    config_path = Path(
        os.getenv("FARCODE_CONFIG", "~/.config/farcode/config.toml")
    ).expanduser()
    if config_path.is_file():
        with config_path.open("rb") as config_file:
            value = tomllib.load(config_file).get("data_root")
        if value:
            return Path(value).expanduser()
    return DEFAULT_DATA_ROOT


def get_adult_data(
    data_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """下载并返回 UCI Adult 数据集的训练/测试特征与标签。

    Args:
        data_root: 数据根目录；未提供时读取 ``FARCODE_DATA_ROOT``，默认使用 ``/content/tmp``。
        返回:
        训练特征、训练标签、测试特征和测试标签。
    """
    root = _data_root(data_root) / "data"
    train_path = root / "adult.data.txt"
    test_path = root / "adult.test.txt"

    download_file(
        url="https://raw.githubusercontent.com/1007530194/data/master/recommendation/data/adult.data.txt",
        path=str(train_path),
    )
    download_file(
        url="https://raw.githubusercontent.com/1007530194/data/master/recommendation/data/adult.test.txt",
        path=str(test_path),
    )

    train_data = pd.read_table(train_path, header=None, delimiter=",")
    test_data = pd.read_table(test_path, header=None, delimiter=",")

    all_columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
        "type",
    ]

    continus_columns = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    dummy_columns = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    train_data["type"] = 1
    test_data["type"] = 2

    all_data = pd.concat([train_data, test_data], axis=0)
    all_data.columns = all_columns

    all_data = pd.get_dummies(all_data, columns=dummy_columns)

    test_data = all_data[all_data["type"] == 2].drop(["type"], axis=1)
    train_data = all_data[all_data["type"] == 1].drop(["type"], axis=1)

    train_data["label"] = train_data["label"].map(
        lambda x: 1 if x.strip() == ">50K" else 0
    )
    test_data["label"] = test_data["label"].map(
        lambda x: 1 if x.strip() == ">50K." else 0
    )

    for col in continus_columns:
        ss = StandardScaler()
        train_data[col] = ss.fit_transform(train_data[[col]].astype(np.float64))
        test_data[col] = ss.transform(test_data[[col]].astype(np.float64))

    train_y = train_data["label"]
    train_x = train_data.drop(["label"], axis=1)
    test_y = test_data["label"]
    test_x = test_data.drop(["label"], axis=1)

    return train_x, train_y, test_x, test_y


class ElectronicsData:
    """下载、转换和构造 Amazon Electronics 数据集。"""

    def __init__(self, data_root: str | Path | None = None) -> None:
        """初始化数据集路径。

        Args:
            data_root: 数据根目录；未提供时读取 ``FARCODE_DATA_ROOT``。
        """
        self.path_root = _data_root(data_root) / "data" / "electronics"

    def download_raw_0(self) -> None:
        """下载原始评论与商品元数据文件并解压。"""
        path_root = self.path_root
        path_root.mkdir(parents=True, exist_ok=True)
        path1 = path_root / "reviews_Electronics_5.json.gz"
        path2 = path_root / "meta_Electronics.json.gz"

        download_file(
            url="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz",
            path=str(path1),
        )

        download_file(
            url="http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz",
            path=str(path2),
        )

        for compressed_path in (path1, path2):
            output_path = compressed_path.with_suffix("")
            with (
                gzip.open(compressed_path, "rb") as source,
                output_path.open("wb") as destination,
            ):
                shutil.copyfileobj(source, destination)

    def convert_pd_1(self) -> None:
        """将原始 JSON 行文件转换为供后续处理的 JSON 文件。"""
        path_root = self.path_root

        def to_df(file_path: Path, required_fields: set[str]) -> pd.DataFrame:
            with file_path.open("r", encoding="utf-8") as fin:
                df = {}
                for line_number, line in enumerate(fin, start=1):
                    try:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            # Amazon 的旧版 meta 文件名虽为 JSON，内容却是 Python 字面量。
                            record = ast.literal_eval(line)
                    except (json.JSONDecodeError, SyntaxError, ValueError) as exc:
                        raise ValueError(
                            f"无法解析 {file_path} 第 {line_number} 行"
                        ) from exc
                    if not isinstance(record, dict):
                        raise TypeError(f"{file_path} 第 {line_number} 行必须是对象")
                    missing = required_fields - record.keys()
                    if missing:
                        fields = ", ".join(sorted(missing))
                        raise ValueError(
                            f"{file_path} 第 {line_number} 行缺少字段: {fields}"
                        )
                    df[line_number - 1] = record
                if not df:
                    raise ValueError(f"{file_path} 不包含任何记录")
                return pd.DataFrame.from_dict(df, orient="index")

        (path_root / "raw_data").mkdir(parents=True, exist_ok=True)
        reviews_df = to_df(
            path_root / "reviews_Electronics_5.json",
            {"reviewerID", "asin", "unixReviewTime"},
        )
        reviews_df.to_json(
            path_root / "raw_data/reviews.json", orient="table", index=False
        )

        meta_df = to_df(path_root / "meta_Electronics.json", {"asin", "categories"})
        meta_df = meta_df[meta_df["asin"].isin(reviews_df["asin"].unique())]
        meta_df = meta_df.reset_index(drop=True)
        meta_df.to_json(path_root / "raw_data/meta.json", orient="table", index=False)

    def remap_id_2(self) -> None:
        """将用户、商品和类目标识映射为连续整数并保存。"""
        random.seed(1234)

        reviews_df = pd.read_json(
            self.path_root / "raw_data/reviews.json", orient="table"
        )[["reviewerID", "asin", "unixReviewTime"]]
        meta_df = pd.read_json(self.path_root / "raw_data/meta.json", orient="table")[
            ["asin", "categories"]
        ]
        meta_df["categories"] = meta_df["categories"].map(lambda x: x[-1][-1])

        def build_map(df, col_name):
            key = sorted(df[col_name].unique().tolist())
            m = dict(zip(key, range(len(key))))
            df[col_name] = df[col_name].map(lambda x: m[x])
            return m, key

        asin_map, asin_key = build_map(meta_df, "asin")
        cate_map, cate_key = build_map(meta_df, "categories")
        revi_map, revi_key = build_map(reviews_df, "reviewerID")

        user_count, item_count, cate_count, example_count = (
            len(revi_map),
            len(asin_map),
            len(cate_map),
            reviews_df.shape[0],
        )
        logger.info(
            f"user_count: {user_count}\t item_count: {item_count}\t "
            f"cate_count: {cate_count}\t example_count: {example_count}"
        )

        meta_df = meta_df.sort_values("asin")
        meta_df = meta_df.reset_index(drop=True)
        reviews_df["asin"] = reviews_df["asin"].map(lambda x: asin_map[x])
        reviews_df = reviews_df.sort_values(["reviewerID", "unixReviewTime"])
        reviews_df = reviews_df.reset_index(drop=True)
        reviews_df = reviews_df[["reviewerID", "asin", "unixReviewTime"]]

        cate_list = [meta_df["categories"][i] for i in range(len(asin_map))]
        cate_list = np.array(cate_list, dtype=np.int32)

        remap_data = {
            "reviews": reviews_df.to_dict(orient="records"),
            "categories": cate_list.tolist(),
            "counts": [user_count, item_count, cate_count, example_count],
            "keys": [asin_key, cate_key, revi_key],
        }
        with (self.path_root / "raw_data/remap.json").open("w", encoding="utf-8") as f:
            json.dump(remap_data, f)

    def init_data(self) -> None:
        """执行下载、转换和标识映射的完整初始化流程。"""
        self.download_raw_0()

        self.convert_pd_1()

        self.remap_id_2()

    def build_dataset(self) -> None:
        """根据用户历史生成训练集和测试集。"""
        path_root = self.path_root
        random.seed(1234)

        with (path_root / "raw_data/remap.json").open(encoding="utf-8") as f:
            remap_data = json.load(f)
        reviews_df = pd.DataFrame(remap_data["reviews"])
        cate_list = remap_data["categories"]
        _user_count, item_count, cate_count, _example_count = remap_data["counts"]

        train_set = []
        test_set = []
        for reviewerID, hist in reviews_df.groupby("reviewerID"):
            pos_list = hist["asin"].tolist()
            if len(pos_list) < 2:
                logger.warning(f"skipping {reviewerID}: fewer than two interactions")
                continue
            available_items = list(set(range(item_count)) - set(pos_list))
            if not available_items:
                raise ValueError(f"no negative item available for user {reviewerID}")

            neg_list = [random.choice(available_items) for _ in range(len(pos_list))]

            for i in range(1, len(pos_list)):
                hist = pos_list[:i]
                if i != len(pos_list) - 1:
                    train_set.append((reviewerID, hist, pos_list[i], 1))
                    train_set.append((reviewerID, hist, neg_list[i], 0))
                else:
                    label = (pos_list[i], neg_list[i])
                    test_set.append((reviewerID, hist, label))

        random.shuffle(train_set)
        random.shuffle(test_set)

        dataset_data = {
            "train": train_set,
            "test": test_set,
            "categories": cate_list,
            "counts": [len(test_set), item_count, cate_count],
        }
        with (path_root / "raw_data/dataset.json").open("w", encoding="utf-8") as f:
            json.dump(dataset_data, f)
