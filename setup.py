from setuptools import find_packages, setup

setup(
    name='cifar_classif',
    packages=find_packages(include=['CIFAR_Classif']),
    version='0.1.0',
    description='A CIFAR-10 image classification project',
    author='Vincent Tardieux',
    install_requires=[],
    setup_requires=['matplotlib==3.7.1', 'numpy==1.24.2', 'opencv-python==4.7.0.72',
                    'pandas==2.0.1', 'scikit-image==0.22.0', 'scikit-learn==1.2.2', 'config'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)