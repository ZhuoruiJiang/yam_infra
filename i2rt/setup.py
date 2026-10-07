"""Package the copied i2rt directory without changing its import layout."""

from setuptools import find_namespace_packages, setup

setup(
    packages=["i2rt", *[f"i2rt.{name}" for name in find_namespace_packages(
        ".", exclude=("build", "build.*", "*.tests", "*.tests.*")
    )]],
    package_dir={"i2rt": "."},
    package_data={"i2rt": ["robot_models/**/*", "flow_base/assets/*"]},
    include_package_data=False,
)
