from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="em_ppca",
        version="1.0",
        description="",
        author="brdav",
        author_email="",
        url="",
        install_requires=["numpy>=1.26.4", "numba>=0.59.1"],
        packages=find_packages(),
    )
