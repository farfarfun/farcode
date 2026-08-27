# funcode

个人深度学习代码片段合集，包含两个互不相关的部分，均为 2019 年前后的学习/实验代码，未做后续维护：

1. `notecode/AttentionDecoder.py`：基于旧版 Keras（`keras.engine.InputSpec`、`keras.layers.recurrent.Recurrent`，对应 Keras 2.x + TensorFlow 1.x 时代的 API）手写实现的一个 Seq2Seq 注意力解码层 `AttentionDecoder`，参考自 Bahdanau et al. 2014《Neural Machine Translation by Jointly Learning to Align and Translate》。
2. `notecode/data/`：两个推荐系统/机器学习经典数据集的下载与预处理脚本——UCI Adult 收入数据集（`get_adult_data`）和亚马逊 Electronics 评论数据集（`ElectronicsData`，用于类似 DIN 的点击率预测场景，含负采样构造训练/测试集的完整流程）。

> 注意：包名/导入名是 `notecode`（历史 `note*` 命名遗留，见 [NAMING.md](https://github.com/farfarfun/todo-list/blob/master/NAMING.md)），与仓库名 `funcode` 不一致。经查 PyPI 上目前**没有**发布 `notecode` 这个包（404）。PyPI 上确实存在一个叫 `funcode`（0.0.1）的包，但经核对其 wheel 内容只有一个空的 `funapi/__init__.py`，是历史上批量占位发布的空包，**和这个仓库的代码毫无关系**，请不要 `pip install funcode` 来使用本仓库的功能。

## 安装

PyPI 上没有可用的发布包，需要从源码安装，且需要自行安装匹配的旧版 `tensorflow`/`keras`（`AttentionDecoder.py` 用的是已废弃的 Keras 2.x `Recurrent` 基类，新版 `tensorflow.keras` 已不兼容）：

```bash
git clone https://github.com/farfarfun/funcode.git
cd funcode
```

## 用法示例

### 注意力解码层

```python
from keras.layers import Input, LSTM, Bidirectional
from keras.models import Model
from notecode.AttentionDecoder import AttentionDecoder

i = Input(shape=(100, 104), dtype='float32')
enc = Bidirectional(LSTM(64, return_sequences=True), merge_mode='concat')(i)
dec = AttentionDecoder(32, 4)(enc)
model = Model(inputs=i, outputs=dec)
model.summary()
```

### 数据集下载/预处理

```python
from notecode.data.data import get_adult_data, ElectronicsData

# UCI Adult 收入数据集，返回标准化后的 train/test 特征和标签
train_x, train_y, test_x, test_y = get_adult_data()

# 亚马逊 Electronics 评论数据集，走完整的下载 -> 转 DataFrame -> id 重映射 -> 构造训练/测试集流程
ed = ElectronicsData()
ed.init_data()
ed.build_dataset()
```

## 已知局限（如实说明）

- 两部分代码之间没有依赖关系，只是放在同一个仓库里的独立学习笔记，不构成一个完整的框架或产品。
- `notecode/data/download.py` 依赖 `pycurl`；数据保存路径硬编码为 Google Colab 风格的 `/content/tmp/`，直接使用需要自行修改。
- `AttentionDecoder.py` 依赖已经停止维护的旧版独立 `keras` 包（非 `tensorflow.keras`），在现代 TensorFlow/Keras 版本下大概率无法直接运行。
