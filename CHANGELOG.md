# 变更日志

## [Unreleased] (续，来自 farfarfun/todo-list#801)

### 修复

- **改名理由不成立，已更正记录**：README 和本文件此前写「`farcode` 用于避开 PyPI 上已有的**无关** `funcode` 占用包」。核对 PyPI 后这句话是错的——[`funcode` 0.0.1](https://pypi.org/project/funcode/)（2023-10-26）的 `Author-email` 是 `1007530194@qq.com`（SPEC §1 列明的本组织邮箱）、`Home-page` 就是 `https://github.com/farfarfun/funcode`，**是本组织自己发的**，不是别人占的名。而且它是个误发产物：包名 `funcode`、`top_level.txt` 里却是 `funapi`、`funapi/__init__.py` 为空、README 标题写着 `funllm`、`console_scripts` 指向 `funget`，从未包含本仓库的代码。按 SPEC §1.4「撞名必须确认是『别人的包』」，这次改名本不该发生。
- **删除 `legacy-dl` extra**（SPEC §12.1：README 的最小示例必须可运行）。`legacy-dl = ["tensorflow>=2.16", "keras>=3.0"]` 装不出能跑 `attention_decoder.py` 的环境：该模块要的 `keras.engine.InputSpec` 和 `keras.layers.recurrent.Recurrent` 自 Keras 2.4 起已删除，Keras 3 连 `keras.engine` 子包都没有；最后提供它们的 Keras 2.2.x/2.3.x + TensorFlow 1.15 只支持到 Python 3.7，与 `requires-python = ">=3.12"` 没有交集——不存在任何可安装的依赖组合。继续声明这个 extra 等于用 `keras>=3.0` 伪装兼容。
- `attention_decoder.py` 的旧 Keras 导入加上保护，失败时抛出说明现状与替代方案的 `ImportError`，而不是一句看不懂的 `No module named 'keras.engine'`；README 把它从「用法示例」降级为「历史参考」，并明确它不是 `farcode` 的公开 API。
- 删掉模块里的死代码 `tfPrint = lambda d, T: tf.Print(...)`：全仓库无人引用，而且 `tf.Print` 在 TensorFlow 2 就已经被移除（改名为 `tf.print`）。它是本模块唯一用到 `tensorflow` 的地方，连带把 `import tensorflow as tf` 也去掉了——现在本模块只依赖 keras。
- `__main__` 自检块里的 `from keras.layers.wrappers import Bidirectional` 在 Keras 2.3 起就不是有效路径（应为 `keras.layers.Bidirectional`），与 README 示例写法也不一致，已对齐。
- `src/farcode/__init__.py` 里 2019 年的 PyCharm 模板头（`@File: __init__.py.py`）换成真正的包文档字符串。

### 新增

- 测试覆盖 `attention_decoder` 的现状（SPEC §12.3）：断言 `import farcode` 不会连带导入它、直接导入会抛出带替代方案的 `ImportError`，且该异常不是裸 `ModuleNotFoundError`。

### 待办（需仓库所有者决定，不在本次改动范围内）

本仓库已改名为 `farcode`，但 `farcode` **从未发布到 PyPI**，而旧的 `funcode` 0.0.1 还挂在上面。SPEC §15.2/§15.3 要求整包改名时发布旧包的最终转发版本，这件事依赖发版，属所有者决定。可选路径：

1. **改回 `funcode`**：改名理由（避开无关占用包）已证伪，而 `funcode` 本就是本组织的名字。改回后直接发 `funcode` 新版本覆盖掉 2023 年那个误发产物，不需要任何转发包。
2. **维持 `farcode`**：先发 `farcode`，再发一个 `funcode` 最终版，其 `dependencies` 指向 `farcode` 做转发，README 首行写明新包名。注意：旧 `funcode` 0.0.1 装出来的顶层包是 `funapi` 不是 `funcode`，`import funcode` 从来就不可用，所以这个转发包实际上没有下游使用者需要照顾。

组织内依赖排查（SPEC §15.2 最后一条）已做：全组织没有任何仓库在 `pyproject.toml` / `requirements.txt` / `setup.py` 里依赖 `funcode`，无需提 PR。

仓库元信息同步更新：`description` 原先把跑不起来的 Attention Decoder 放在第一位，改为先说数据集下载/预处理这个真正可用的部分；topics 去掉 `tensorflow`、`keras`（仓库里已经不再依赖这两个包），改为 `dataset`/`data-preprocessing`/`machine-learning`/`recommender-system`，共 7 个（SPEC §11 要求 5-8 个）。

另：仓库 `homepage` 原指向 `https://pypi.org/project/funcode/`（那个页面描述的是 2023 年的误发产物，不是本仓库），按 SPEC §11「仅在包确实已发布且确属本组织时才设」已清空；`farcode` 发布后再设为 `https://pypi.org/project/farcode/`。`todo-list` 的 `NAMING.md`「三者一致」名单里仍写着 `funcode`，需随上述决定一并更新。

## [Unreleased]

### 新增

- 新增 `farcode` 发布名和对应源码包（当时的理由「避开 PyPI 上无关的 `funcode` 占用包」事后证实不成立，见本文件顶部 #801 一节）。
- 新增函数参数、`FARCODE_DATA_ROOT` 环境变量和 TOML 配置形式的数据目录配置。

### 修复

- 源码迁移到标准 `src/farcode/` 布局，同步更新 `pyproject.toml` 的 hatch 打包配置。
- `AttentionDecoder.py` 重命名为 `attention_decoder.py`（模块名 snake_case），类名 `AttentionDecoder` 不变。
- `pyproject.toml` 补齐运行时依赖（numpy/pandas/scikit-learn/funget/funshell/farlog）及版本下限；为 `tensorflow`/`keras` 补充兼容版本约束并拆分到 `legacy-dl` extra；`requires-python` 提升到 `>=3.12` 以匹配 `funget` 的要求；提交 `uv.lock`。
- `download.py` 改用 `funget.simple_download` 替代自实现的 `pycurl` 下载；`data.py` 的 `os.system` gzip 调用改用 `funshell.run_shell` 并在失败时抛出异常，不再吞掉退出码。
- 移除各处 `print()`，改用 `farlog.getLogger`。
- README 补充 `uv sync` 安装命令与组织介绍固定区块；示例代码同步更新模块路径。
- 删除基于 `setup.py`/`twine`/自动 `git commit&push` 的遗留 `build.sh`（仓库中并无 `setup.py`，构建发布统一走 `funbuild`）。
- 补充 `farcode.data` 公开 API 的导入型测试用例。

### 变更

- **不兼容变更：**项目发布名和导入名从 `funcode` 改为 `farcode`，使用者需将 `import funcode` 改为 `import farcode`。

### 废弃

- `funcode` 名称不再用于本项目。（当时还写了「旧名称的兼容转发包不发布」，这违反 SPEC §15.3；处理方式见本文件顶部 #801 一节的「待办」。）

## [Unreleased] (续，来自 farfarfun/todo-list#656)

### 修复

- `pyproject.toml` 的 `farlog` 依赖下限从 `>=1.1.3` 提升到 `>=1.1.7`，避免解析到过旧版本。
- `ElectronicsData.convert_pd_1()` 不再用 `eval()` 执行下载内容，改为 `json.loads()` 解析，
  对旧版 Amazon meta 文件内容回退到 `ast.literal_eval()`；校验记录为对象且包含必需字段，
  出错时抛出带文件路径和行号的异常。
- `AttentionDecoder` 类及其 `call()` 方法补齐中文 docstring，说明输入形状、参数、返回值
  和 `return_probabilities` 行为。

### 新增

- 补充 `farcode.data` 公开 API 的行为测试：下载跳过/失败、数据根目录解析、
  `get_adult_data` 正常路径、`convert_pd_1` 的 JSON/旧版字面量解析与空文件/
  非法行/缺字段错误路径。
