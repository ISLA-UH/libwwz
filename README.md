# libwwz

### Description

This library provides functions for computing the Weighted Wavelet Z-Transform using Python v3.6+.

The code is based on [Foster's](http://adsabs.harvard.edu/full/1996AJ....112.1709F) and [Templeton's](http://adsabs.harvard.edu/full/2004JAVSO..32...41T) Fortran code as well as [eaydin's Python 2.7 WWZ library](https://github.com/eaydin/WWZ). The code is updated to use modern numpy methods and allow for float value tau.

Specific equations can be found on Grant Foster's [_Wavelets for period analysis of unevenly sampled time series_](http://adsabs.harvard.edu/full/1996AJ....112.1709F).   

### Installation

#### Install with pip

```
pip install libwwz
```

#### Install from source

### Usage

### Development

Contributions are welcome.

If you would like to contribute:

* Open a new issue on BitBucket
* Fork and then clone this repository (https://bitbucket.org/redvoxhi/libwwz/src/master/)
* Install the development requirements in `dev_requirements.txt`
* Create a new branch in your forked repo following the format of `issue-[issue number]`
* Make your modifications, commit, and push to your forked repository
* Create a pull request

Before you commit:

* Follow Python's [pep8 guidelines](https://www.python.org/dev/peps/pep-0008/)
* Create unit tests for changes
* Ensure that all current tests pass by running the `test.sh` script
* Lint the code using pylint, do not make a pull request until there are no linting errors
* Lint the code using mypy, do not make a pull request until all typing errors are resolved