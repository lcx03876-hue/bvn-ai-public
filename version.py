"""Single source of truth for the project version.

Kept in Python rather than a plain ``VERSION`` text file so that any script can
do ``from version import __version__`` without touching the filesystem.

The version follows Semantic Versioning (https://semver.org):

    MAJOR  incompatible changes to the tooling interfaces
    MINOR  new capability, backwards compatible
    PATCH  backwards-compatible fixes

While the major version is 0, the observation layout, action set, CLI flags and
run-directory format may still change between minor releases.

Release history lives in ``CHANGELOG.md``; releases are tagged in git as
``v<version>``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]


def version_string() -> str:
    """Human-readable version, for banners and ``--version`` flags."""
    return f"bvn-ai {__version__}"


if __name__ == "__main__":
    print(version_string())
