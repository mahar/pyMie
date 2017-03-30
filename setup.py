from distutils.core import setup

setup(
    name='pyMie',
    version='2.0.0',
    author='Ch. Mavidis',
    author_email='mavidis@iesl.forth.gr',
    packages=['pyMie'],
    package_dir={'pyMie': '.'},
    url='',
    license='LICENSE.txt',
    description='pyMie calculates the extinction, scattering and absorption efficiencies for a cylinder immersed in a host.',
    long_description=open('README.txt').read(),
    install_requires=["numpy","scipy"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2"],
)