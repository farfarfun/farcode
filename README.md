# farcode

个人深度学习代码片段合集，包含两个互不相关的部分，均为 2019 年前后的学习/实验代码，未做后续维护：

1. `farcode/data/`（**本包唯一可用的部分**）：两个推荐系统/机器学习经典数据集的下载与预处理脚本——UCI Adult 收入数据集（`get_adult_data`）和亚马逊 Electronics 评论数据集（`ElectronicsData`，用于类似 DIN 的点击率预测场景，含负采样构造训练/测试集的完整流程）。
2. `farcode/attention_decoder.py`（**历史参考，无法运行**）：Keras 2.2 + TensorFlow 1.x 时代手写的 Seq2Seq 注意力解码层 `AttentionDecoder`，参考自 Bahdanau et al. 2014《Neural Machine Translation by Jointly Learning to Align and Translate》。详见下面的「关于 `attention_decoder.py`」。

## 安装

PyPI 上没有可用的发布包，需要从源码安装：

```bash
git clone https://github.com/farfarfun/farcode.git
cd farcode
uv sync
```

数据集下载/预处理部分（`farcode.data`）的依赖已在 `pyproject.toml` 中声明，`uv sync` 后即可用。数据目录可通过函数参数、`FARCODE_DATA_ROOT` 环境变量或 `~/.config/farcode/config.toml` 中的 `data_root` 配置。

## 用法示例

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

## 关于 `attention_decoder.py`

这个模块**在今天没有任何可安装的依赖组合能让它跑起来**，因此它不是 `farcode` 的公开 API，也没有给出用法示例：

- 它需要 `keras.engine.InputSpec` 和 `keras.layers.recurrent.Recurrent`。这两个符号自 **Keras 2.4 起已被删除**，Keras 3 连 `keras.engine` 这个子包都没有。
- 最后提供它们的是 **Keras 2.2.x / 2.3.x 配 TensorFlow 1.15**，而那套组合只支持到 **Python 3.7**，与本项目的 `requires-python = ">=3.12"` 没有交集。
- 所以不存在「装上某个 extra 就能用」的说法。此前 `pyproject.toml` 里的 `legacy-dl = ["tensorflow>=2.16", "keras>=3.0"]` 是个会误导人的声明——装完照样 `ImportError`——现已删除。

`import farcode` 不会导入这个模块；直接 `import farcode.attention_decoder` 会得到一条说明上述现状的 `ImportError`，而不是一句看不懂的 `No module named 'keras.engine'`。源码保留下来只作为算法实现的历史参考。需要注意力机制请用 `keras.layers.MultiHeadAttention`，或用 `keras.layers.RNN` + 自定义 Cell 按 Keras 3 的接口重写。

## 已知局限（如实说明）

- 两部分代码之间没有依赖关系，只是放在同一个仓库里的独立学习笔记，不构成一个完整的框架或产品。
- 数据默认保存到 Google Colab 风格的 `/content/tmp/`，可通过配置覆盖。
- 本项目尚未发布到 PyPI。PyPI 上的 [`funcode` 0.0.1](https://pypi.org/project/funcode/)（2023-10-26）是本组织自己误发的一个产物——包名叫 `funcode`、顶层包却是空的 `funapi`、README 标题写着 `funllm`、入口点指向 `funget`，从未包含本仓库的代码，`import funcode` 在任何版本下都没有用过。它不是别人占的名。本仓库改名为 `farcode` 的理由（「避开无关占用包」）因此并不成立，是否发布 `farcode`、是否清理旧 `funcode`、以及要不要改回 `funcode`，见 CHANGELOG 的「待办（需仓库所有者决定）」。

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
