# funcode

个人深度学习代码片段合集，包含两个互不相关的部分，均为 2019 年前后的学习/实验代码，未做后续维护：

1. `funcode/attention_decoder.py`：基于旧版 Keras（`keras.engine.InputSpec`、`keras.layers.recurrent.Recurrent`，对应 Keras 2.x + TensorFlow 1.x 时代的 API）手写实现的一个 Seq2Seq 注意力解码层 `AttentionDecoder`，参考自 Bahdanau et al. 2014《Neural Machine Translation by Jointly Learning to Align and Translate》。
2. `funcode/data/`：两个推荐系统/机器学习经典数据集的下载与预处理脚本——UCI Adult 收入数据集（`get_adult_data`）和亚马逊 Electronics 评论数据集（`ElectronicsData`，用于类似 DIN 的点击率预测场景，含负采样构造训练/测试集的完整流程）。

> 注意：PyPI 上已存在一个同名的 `funcode`（0.0.1）包，但经核对其 wheel 内容只有一个空的 `funapi/__init__.py`，是历史上批量占位发布的空包，**和这个仓库的代码毫无关系**，请不要 `pip install funcode` 来使用本仓库的功能。本仓库尚未发布到 PyPI，与该同名包的命名冲突如何处理（申请占用方协商 / 改用 `far*` 前缀）留待仓库所有者决定。

## 安装

PyPI 上没有可用的发布包，需要从源码安装：

```bash
git clone https://github.com/farfarfun/funcode.git
cd funcode
uv sync
```

数据集下载/预处理部分（`funcode.data`）的依赖已在 `pyproject.toml` 中声明，`uv sync` 后即可用。注意力解码层额外需要旧版 `tensorflow`/`keras`（`attention_decoder.py` 用的是已废弃的 Keras 2.x `Recurrent` 基类，新版 `tensorflow.keras` 已不兼容），可通过 `uv sync --extra legacy-dl` 安装，但不保证在现代版本下可运行。

## 用法示例

### 注意力解码层

```python
from keras.layers import Input, LSTM, Bidirectional
from keras.models import Model
from funcode.attention_decoder import AttentionDecoder

i = Input(shape=(100, 104), dtype='float32')
enc = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat')(i)
dec = AttentionDecoder(32, 4)(enc)
model = Model(inputs=i, outputs=dec)
model.summary()
```

### 数据集下载/预处理

```python
from funcode.data.data import get_adult_data, ElectronicsData

# UCI Adult 收入数据集，返回标准化后的 train/test 特征和标签
train_x, train_y, test_x, test_y = get_adult_data()

# 亚马逊 Electronics 评论数据集，走完整的下载 -> 转 DataFrame -> id 重映射 -> 构造训练/测试集流程
ed = ElectronicsData()
ed.init_data()
ed.build_dataset()
```

## 已知局限（如实说明）

- 两部分代码之间没有依赖关系，只是放在同一个仓库里的独立学习笔记，不构成一个完整的框架或产品。
- `funcode/data/download.py` 数据保存路径硬编码为 Google Colab 风格的 `/content/tmp/`，直接使用需要自行修改。
- `attention_decoder.py` 依赖已经停止维护的旧版独立 `keras` 包（非 `tensorflow.keras`），在现代 TensorFlow/Keras 版本下大概率无法直接运行。

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
