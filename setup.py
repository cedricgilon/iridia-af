import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

a = setuptools.find_packages()
print("-"*100)
print(a)
print("-"*100)


setuptools.setup(
    name="iridia_af",
    version="1.0.0",
    author="Cédric Gilon",
    author_email="cedric.gilon@ulb.be",
    description="Utils tools for IRIDIA AF database",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cedricgilon/iridia-af",
    packages=setuptools.find_packages(),
    install_requires=[
        "h5py",
        "numpy",
        "pandas",
        "matplotlib",
        "hrv-analysis"
        ]
)

