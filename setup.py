import os 
from setuptools import setup, find_packages


#Use README file for long description of the project
##     RootDir
###    - README.md
###    - setup.py

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def parse_requiremetns(fname):
    with open(fname) as f:
        required = f.read().splitlines()
    return required
    
setup(
    name = "ContinuousControlAgent",
    author = "Akhil Singh Rana",
    author_email = "er.akhil.singh.rana@gmail.com",
    description = ("This is project done at Airbus Defence and Space"
                    "Jammer Detection and Classification using deepLearning"),
    long_description = read("README.md"),
    install_requires = parse_requiremetns("requirements.txt"),    
    packages=find_packages(),
)