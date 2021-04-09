import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fr:
    requirements = fr.read().splitlines()

setuptools.setup(
    name="invertpy",
    version="1.0.1",
    author="Evripidis Gkanias",
    maintainer="Evripidis Gkanias",
    author_email="ev.gkanias@ed.ac.uk",
    maintainer_email="ev.gkanias@ed.ac.uk",
    description="Python package of computational models relevant to invertebrate processing, from the environment to"
                "sensor responses and to deeper neural responses in the invertebrate brain.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evgkanias/InvertBrain",
    license="GPLv3+",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Licence :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent"
    ],
    package_dir={"": "src/"},
    packages=["invertpy", "invertpy.brain", "invertpy.io", "invertpy.sense"],
    python_requires=">=3.8",
    install_requires=requirements
)
