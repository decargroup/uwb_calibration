import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyuwbcalib",
    version="1.0.0",
    author="Mohammed Shalaby, Charles Cossette",
    author_email="mohammed.shalaby@mail.mcgill.ca, charles.cossette@mail.mcgill.ca",
    description="A package for everything calibration related for UWB modules.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="<>",
    packages=setuptools.find_packages(),
    install_requires=[
        'bagpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
