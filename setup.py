from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT="-e ."  
#whenever we add a package, this will automatically trigger setup.py file building the package. 
"""whenever we run requirements.txt -e . will come in setup.py file but this should not come.
to resolve this issue we will add a code in setup.py which we skip this."""

def get_requirements(file_path:str)->List[str]:
    """
    this function will return the list of requirements
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='my_ml_project',  # Replace with your project name
    version='1.0',  # Replace with your project version
    packages=find_packages(),  # Automatically discover and include all packages
    install_requires=get_requirements("requirements.txt"),
    author='Ravina',  # Replace with your name
    author_email='vermaravina029@gmail.com',  # Replace with your email
    description='end to end machine learing project',
    url="https://github.com/ravina029/first_end_to_end.git" # Replace with your project's GitHub URL
)
