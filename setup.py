from setuptools import setup, find_packages
import os
from glob import glob

package_name = "vln_project"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            [f"resource/{package_name}"]),
        (f"share/{package_name}",
            ["package.xml"]),
        (f"share/{package_name}/launch",
            glob("launch/*.py")),
        (f"share/{package_name}/config",
            glob("config/*.yaml")),
        (f"share/{package_name}/worlds",
            glob("worlds/*.world")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gaurav",
    description="VLN with fine-tuned CLIP + TurtleBot3 waffle + Nav2",
    entry_points={
        "console_scripts": [
            "object_detector_node   = vln_project.object_detector_node:main",
            "frontier_explorer_node = vln_project.frontier_explorer_node:main",
            "vln_command_node       = vln_project.vln_command_node:main",
        ],
    },
)
