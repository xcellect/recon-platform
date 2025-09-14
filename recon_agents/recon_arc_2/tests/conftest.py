import os


def pytest_ignore_collect(path, config):
    try:
        base = os.path.basename(str(path))
        # Only collect gate-focused ARC3 tests
        if base.startswith("test_") and not base.startswith("test_arc3_gates"):
            return True
    except Exception:
        return False
    return False


