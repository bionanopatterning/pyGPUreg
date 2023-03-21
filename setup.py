from setuptools import setup, find_packages

setup(
    name='pyGPUreg',
    version='0.0.9',
    license='GPLv3',
    author="Mart G.F. Last",
    author_email='m.g.f.last@lumc.nl',
    description='GPU-accelerated image registration.\ngithub.com/bionanopatterning/pyGPUfit',
    packages=find_packages(),
    package_data={'pyGPUreg': ['*.glsl', 'shaders/*glsl']},
    include_package_data=True,
    install_requires=[
        "numpy>=1.3.0",
        "PyOpenGL>=3.1.6",
        "glfw>=2.5.5"
      ],
)