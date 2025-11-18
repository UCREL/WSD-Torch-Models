import re

import wsd_torch_models


def test_version() -> None:
    version = wsd_torch_models.__version__
    assert isinstance(version, str)
    assert re.search(r"^\d+\.\d+\.\d+$", version) is not None
