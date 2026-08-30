# NOTE: This test is intentionally minimal.
#
# funcode's pyproject.toml declares zero runtime dependencies, and its
# top-level __init__.py is empty, so `import funcode` is safe on its own.
# Its submodules (funcode.data.data, funcode.AttentionDecoder) import
# heavy, undeclared ML dependencies (tensorflow, keras, scikit-learn,
# pandas, numpy) left over from 2019-era experiments and would fail to
# import in a clean environment. That's a pre-existing issue in funcode's
# own source, out of scope for this smoke test, so only the top-level
# package import is exercised here.
import funcode


def test_import_funcode():
    assert funcode is not None
