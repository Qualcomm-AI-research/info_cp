# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


from distutils.core import setup

from setuptools import find_packages

extra_files = []
scripts = ["scripts/run_conformal_training.py"]
package_list = find_packages()

setup(
    name="info_cp",
    packages=package_list,
    scripts=scripts,
    package_data={"": extra_files},
    include_package_data=True,
    python_requires=">=3.8",
)
