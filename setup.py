import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras-regularization",
    version="1.0.0",
    author="Mohammad Mahdi Bejani",
    author_email="mbejani@aut.ac.ir",
    description="regularization",
    long_description=long_description,
    long_description_content_type="README!",
    url="https://github.com/mmbejani/keras-regularization/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
