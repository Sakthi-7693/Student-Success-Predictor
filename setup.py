from setuptools import find_packages,setup
from typing import List

e_dot = '-e .'

def get_requirements(file_path:str)->List[str]:
    "This function will return the list of requirements"
    requirements=[]
    #Writing a for loop
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.strip() for req in file_obj.readlines()]

        if e_dot in requirements:
            requirements.remove(e_dot)
    return requirements



setup(
    name='Student-Success-Predictor',
    version='0.0.1',
    author='Sakthi',
    author_email='sakthi.1617s@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirement.txt')
)
