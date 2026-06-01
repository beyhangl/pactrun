# Enable pytest's `pytester` fixture so the pactrun pytest-plugin can be tested
# in isolated sub-runs (see tests/test_pytest_plugin.py).
pytest_plugins = ["pytester"]
