# @Time    : 2019/03/30 12:10
# @Author  : niuliangtao
# @Site    :
# @File    : data.py
# @Software: PyCharm
import os
import pickle
import random
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from farlog import getLogger
from funshell import run_shell
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

        cmd1 = f'cd "{path_root}" && gzip -d reviews_Electronics_5.json.gz'
        cmd2 = f'cd "{path_root}" && gzip -d meta_Electronics.json.gz'

        for cmd in (cmd1, cmd2):
            code = run_shell(cmd)
            if code != "0":
                raise RuntimeError(f"command failed (exit {code}): {cmd}")

    def convert_pd_1(self) -> None:
        """将原始 JSON 行文件转换为供后续处理的 pickle 文件。"""
        path_root = self.path_root

        def to_df(file_path):
            with open(file_path, "r") as fin:
                df = {}
                for i, line in enumerate(fin):
                    df[i] = eval(line)
                df = pd.DataFrame.from_dict(df, orient="index")
                return df

        (path_root / "raw_data").mkdir(parents=True, exist_ok=True)
        reviews_df = to_df(path_root / "reviews_Electronics_5.json")
        with open(path_root / "raw_data/reviews.pkl", "wb") as f:
            pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

        meta_df = to_df(path_root / "meta_Electronics.json")
        meta_df = meta_df[meta_df["asin"].isin(reviews_df["asin"].unique())]
        meta_df = meta_df.reset_index(drop=True)
        with open(path_root / "raw_data/meta.pkl", "wb") as f:
            pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)

    def remap_id_2(self) -> None:
        """将用户、商品和类目标识映射为连续整数并保存。"""
        random.seed(1234)

        with open(self.path_root / "raw_data/reviews.pkl", "rb") as f:
            reviews_df = pickle.load(f)
            reviews_df = reviews_df[["reviewerID", "asin", "unixReviewTime"]]
        with open(self.path_root / "raw_data/meta.pkl", "rb") as f:
            meta_df = pickle.load(f)
            meta_df = meta_df[["asin", "categories"]]
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

        with open(self.path_root / "raw_data/remap.pkl", "wb") as f:
            pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)  # uid, iid
            pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)  # cid of iid line
            pickle.dump(
                (user_count, item_count, cate_count, example_count),
                f,
                pickle.HIGHEST_PROTOCOL,
            )
            pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)

    def init_data(self) -> None:
        """执行下载、转换和标识映射的完整初始化流程。"""
        self.download_raw_0()

        self.convert_pd_1()

        self.remap_id_2()

    def build_dataset(self) -> None:
        """根据用户历史生成训练集和测试集。"""
        path_root = self.path_root
        random.seed(1234)

        with open(path_root / "raw_data/remap.pkl", "rb") as f:
            reviews_df = pickle.load(f)
            cate_list = pickle.load(f)
            user_count, item_count, cate_count, _example_count = pickle.load(f)

        train_set = []
        test_set = []
        for reviewerID, hist in reviews_df.groupby("reviewerID"):
            pos_list = hist["asin"].tolist()

            def gen_neg(items: list[int] = pos_list) -> int:
                neg = items[0]
                while neg in items:
                    neg = random.randint(0, item_count - 1)
                return neg

            neg_list = [gen_neg() for i in range(len(pos_list))]

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

        assert len(test_set) == user_count
        # assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

        with open(path_root / "raw_data/dataset.pkl", "wb") as f:
            pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(
                (user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL
            )
