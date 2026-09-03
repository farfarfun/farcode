# Changelog

## [Unreleased]

### 修复

- 源码迁移到标准 `src/funcode/` 布局，同步更新 `pyproject.toml` 的 hatch 打包配置。
- `AttentionDecoder.py` 重命名为 `attention_decoder.py`（模块名 snake_case），类名 `AttentionDecoder` 不变。
- `pyproject.toml` 补齐运行时依赖（numpy/pandas/scikit-learn/funget/funshell/farlog）及版本下限；`tensorflow`/`keras` 拆分到 `legacy-dl` extra；`requires-python` 提升到 `>=3.12` 以匹配 `funget` 的要求；提交 `uv.lock`。
- `download.py` 改用 `funget.simple_download` 替代自实现的 `pycurl` 下载；`data.py` 的 `os.system` gzip 调用改用 `funshell.run_shell` 并在失败时抛出异常，不再吞掉退出码。
- 移除各处 `print()`，改用 `farlog.getLogger`。
- README 补充 `uv sync` 安装命令与组织介绍固定区块；示例代码同步更新模块路径。
- 删除基于 `setup.py`/`twine`/自动 `git commit&push` 的遗留 `build.sh`（仓库中并无 `setup.py`，构建发布统一走 `funbuild`）。
- 补充 `funcode.data` 公开 API 的导入型测试用例。

### Changed

- **Breaking:** renamed the import name and (nominal) PyPI package name from `notecode` to
  `funcode` to match the repository name. Anyone doing `import notecode` must switch to
  `import funcode`.
- Note: unlike other `note*` -> `fun*` renames in this org, `notecode` was never actually
  published to PyPI (404 before this change), so there is no old `notecode` package to give a
  final forwarding release to. Separately, PyPI already has an unrelated, pre-existing empty
  placeholder package also named `funcode` (0.0.1, contains only an empty `funapi/__init__.py`)
  that is not owned by/derived from this repo. Publishing this project to PyPI as `funcode`
  will require the repo owner to resolve that naming collision manually (e.g. requesting the
  name or picking a different distribution name) — not done as part of this rename.
