from setuptools import setup, find_packages

setup(
    name='mod-flashzoi',  # Changed
    version='0.4.3',
    author='Johannes Hingerl',
    author_email='johannes.hingerl@tum.de',
    packages=['mod_flashzoi'],  # Changed
    # packages = find_packages(exclude=[]),
    include_package_data = True,
    url='https://github.com/khorms/mod_flashzoi',
    license='LICENSE',
    description='The Borzoi model from Linder et al., but in Pytorch - Modified version',
    install_requires=[
        "einops >= 0.5",
        "numpy >= 1.14.2",
        "torch >= 2.1.0",
        "transformers >= 4.34.1,<4.51.0",
        "jupyter >= 1.0.0; extra == 'dev'",
    "intervaltree~=3.1.0",
    "pandas",
        "flash-attn >= 2.6.3; extra == 'flash'"
    ],
)