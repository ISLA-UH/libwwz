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

* Clone this repository from https://github.com/RedVoxInc/libwwz/
* Run `pip3 install . --upgrade --no-cache` from the project root

### Usage

The `wwt` function is available after importing `libwwz`.

```
import libwwz

result: np.ndarray = libwwz.wwt(...)
```

### Examples

Full examples can be found in the [examples directory](https://github.com/RedVoxInc/libwwz/blob/master/examples/example_wwz.py).

### API Documentation

API documentation is available at: https://redvoxinc.github.io/libwwz/

### Contributors and Funding

This project was funded by Consortium of Enabling Technologies and Innovation (ETI) under 
the Department of Energy (DOE) through University of Hawaiʻi.

I would like to thank my advisor Milton A. Garcés, Ph.D. for the continuing support and advice on this code.


### Development

Contributions are welcome.

If you would like to contribute:

* Open a new issue on BitBucket
* Fork and then clone this repository (https://github.com/RedVoxInc/libwwz/)
* Install the development requirements in `dev_requirements.txt` with `pip install -r dev_requirements.txt`
* Create a new branch in your forked repo following the format of `issue-[issue number]`
* Make your modifications, commit, and push to your forked repository
* Create a pull request

Before you commit:

* Follow Python's [pep8 guidelines](https://www.python.org/dev/peps/pep-0008/)
* Create unit tests for changes
* Ensure that all current tests pass by running the `run_tests.sh` script
* Lint the code using [pylint](https://www.pylint.org/) by running the `run_pylint.sh` script, do not make a pull request until there are no linting errors
* Statically check types using [mypy](http://mypy-lang.org/) by running the `run_mypy.sh` script, do not make a pull request until all typing errors are resolved

Always remember [The Zen of Python](https://www.python.org/dev/peps/pep-0020/)

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```
