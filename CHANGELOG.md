# 变更日志

## [Unreleased]

### 新增

- 新增 `farcode` 发布名和对应源码包，避开 PyPI 上无关的 `funcode` 占用包。
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

- `funcode` 名称不再用于本项目；旧名称的兼容转发包不发布。
