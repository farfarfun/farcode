# Changelog

## [Unreleased]

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
