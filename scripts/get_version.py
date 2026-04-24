import argparse
import io
import sys

if sys.version_info.major == 3 and sys.version_info.minor < 11:
    raise RuntimeError(f"Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}")

import tomllib

if __name__ == "__main__":
    description = (
        "prints to stdout the version number of the given pyproject.toml file"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("toml_file_path",
                        type=argparse.FileType(mode="r", encoding="utf-8"),
                        help="File path to the pyproject.toml file")
    args = parser.parse_args()
    toml_file_io = args.toml_file_path
    assert isinstance(toml_file_io, io.TextIOWrapper)
    print(tomllib.loads(toml_file_io.read())['project']['version'])