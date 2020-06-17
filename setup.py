from setuptools import setup, find_packages

with open("requirements.txt", "r") as requirements_file:
    requirements = list(map(lambda line: line.strip(), requirements_file.readlines()))
    requirements = list(filter(lambda line: (not line.startswith("#")) and len(line) > 0, requirements))

setup(name="libwwz",
      version = "1.0.0",
      url='https://bitbucket.org/redvoxhi/libwwz/src/master/',
      license='Apache',
      author='RedVox',
      author_email='dev@redvoxsound.com',
      description='Library for computing the weighted wavelet Z transform.',
      packages=find_packages(include=[
            "libwwz",
            "beta_wwz"
      ],
                             exclude=['tests']),
      long_description=open('README.md').read(),
      install_requires=requirements,
      python_requires=">=3.6")
