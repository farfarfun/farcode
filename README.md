# farcode

个人深度学习代码片段合集，包含两个互不相关的部分，均为 2019 年前后的学习/实验代码，未做后续维护：

1. `farcode/attention_decoder.py`：基于旧版 Keras（`keras.engine.InputSpec`、`keras.layers.recurrent.Recurrent`，对应 Keras 2.x + TensorFlow 1.x 时代的 API）手写实现的一个 Seq2Seq 注意力解码层 `AttentionDecoder`，参考自 Bahdanau et al. 2014《Neural Machine Translation by Jointly Learning to Align and Translate》。
2. `farcode/data/`：两个推荐系统/机器学习经典数据集的下载与预处理脚本——UCI Adult 收入数据集（`get_adult_data`）和亚马逊 Electronics 评论数据集（`ElectronicsData`，用于类似 DIN 的点击率预测场景，含负采样构造训练/测试集的完整流程）。

> 本项目发布名和源码包名为 `farcode`，用于避开 PyPI 上已有的无关 `funcode` 占用包。

## 安装

PyPI 上没有可用的发布包，需要从源码安装：

```bash
git clone https://github.com/farfarfun/farcode.git
cd farcode
uv sync
```

数据集下载/预处理部分（`farcode.data`）的依赖已在 `pyproject.toml` 中声明，`uv sync` 后即可用。注意力解码层额外需要旧版 `tensorflow`/`keras`（`attention_decoder.py` 用的是已废弃的 Keras 2.x `Recurrent` 基类，新版 `tensorflow.keras` 已不兼容），可通过 `uv sync --extra legacy-dl` 安装，但不保证在现代版本下可运行。数据目录可通过函数参数、`FARCODE_DATA_ROOT` 环境变量或 `~/.config/farcode/config.toml` 中的 `data_root` 配置。

## 用法示例

### 注意力解码层

```python
from keras.layers import Input, LSTM, Bidirectional
from keras.models import Model
from farcode.attention_decoder import AttentionDecoder

i = Input(shape=(100, 104), dtype="float32")
enc = Bidirectional(LSTM(64, return_sequences=True), merge_mode="concat")(i)
dec = AttentionDecoder(32, 4)(enc)
model = Model(inputs=i, outputs=dec)
model.summary()
```

### 数据集下载/预处理

```python
from farcode.data.data import get_adult_data, ElectronicsData

# UCI Adult 收入数据集，返回标准化后的 train/test 特征和标签
train_x, train_y, test_x, test_y = get_adult_data()

# 亚马逊 Electronics 评论数据集，走完整的下载 -> 转 DataFrame -> id 重映射 -> 构造训练/测试集流程
ed = ElectronicsData()
ed.init_data()
ed.build_dataset()
```

## 已知局限（如实说明）

- 两部分代码之间没有依赖关系，只是放在同一个仓库里的独立学习笔记，不构成一个完整的框架或产品。
- 数据默认保存到 Google Colab 风格的 `/content/tmp/`，可通过配置覆盖。
- `attention_decoder.py` 依赖已经停止维护的旧版独立 `keras` 包（非 `tensorflow.keras`），在现代 TensorFlow/Keras 版本下大概率无法直接运行。

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
