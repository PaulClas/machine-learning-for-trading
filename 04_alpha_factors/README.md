## Chapter 04: Alpha Factor Research & Evaluation

This section introduces the algorithmic trading simulator [`zipline`](http://www.zipline.io/index.html) and the [`alphalens`](http://quantopian.github.io/alphalens/) library for the performance analysis of predictive (alpha) factors.

### `zipline` Installation

The following is an excerpt from the `zipline` installation instructions. See [docs](http://www.zipline.io/index.html) for more detail.


#### Installing With ``pip``


Assuming you have all required (see note below) non-Python dependencies, you can install Zipline with ``pip`` via:


    $ pip install zipline

**Note:** Installing Zipline via ``pip`` is slightly more involved than the average Python package.  Simply running ``pip install zipline`` will likely fail if you've never installed any scientific Python packages before.

There are two reasons for the additional complexity:

1. Zipline ships several C extensions that require access to the CPython C API.
   In order to build the C extensions, ``pip`` needs access to the CPython
   header files for your Python installation.

2. Zipline depends on [`numpy`](http://www.numpy.org/>), the core library for
   numerical array computing in Python.  Numpy depends on having the [`LAPACK`](http://www.netlib.org/lapack) linear algebra routines available.

Because LAPACK and the CPython headers are binary dependencies, the correct way to install them varies from platform to platform.  On Linux, users generally acquire these dependencies via a package manager like `apt`, `yum`, or `pacman`.  On OSX, `[Homebrew](http://www.brew.sh) is a popular choice providing similar functionality.

See the full [Zipline Install Documentation](https://github.com/quantopian/zipline#installation) for more information on acquiring binary dependencies for your specific platform.

#### Installing with `conda`

Another way to install Zipline is via the `conda` package manager, which comes as part of [`Anaconda`](http://continuum.io/downloads) or can be installed via `pip install conda`.

Once set up, you can install Zipline from the `Quantopian` channel:

    $ conda install -c Quantopian zipline

Currently supported platforms include:

-  GNU/Linux 64-bit
-  OSX 64-bit
-  Windows 64-bit

.. note::

   Windows 32-bit may work; however, it is not currently included in
   continuous integration tests.


### `alphalens` Installation

The following is an excerpt from the `alphalens` installation instructions. See [docs](http://quantopian.github.io/alphalens/) for more detail.

Install with pip:

    pip install alphalens

Install with conda:

    conda install -c conda-forge alphalens

Install from the master branch of the Alphalens repository (development code):

    pip install git+https://github.com/quantopian/alphalens

Alphalens depends on:

-  [`matplotlib`]( <https://github.com/matplotlib/matplotlib)
-  [`numpy`](https://github.com/numpy/numpy)
-  [`pandas`](https://github.com/pydata/pandas)
-  [`scipy`](https://github.com/scipy/scipy)
-  [`seaborn`](https://github.com/mwaskom/seaborn)
-  [`statsmodels`](https://github.com/statsmodels/statsmodels)

## References

### Empirical research on pricing anomalies and risk factors

- "Dissecting Anomalies" by Eugene Fama and Ken French (2008)
- "Explaining Stock Returns: A Literature Review" by James L. Davis (2001)
- "Market Efficiency, Long-Term Returns, and Behavioral Finance" by Eugene Fama (1997)
- "The Efficient Market Hypothesis and It's Critics" by Burton Malkiel (2003)
- "The New Palgrave Dictionary of Economics" (2008) by Steven Durlauf and Lawrence Blume, 2nd ed.
- "Anomalies and Market Efficiency" by G. William Schwert25 (Ch. 15 in Handbook of the
- "Economics of Finance", by Constantinides, Harris, and Stulz, 2003)
- "Investor Psychology and Asset Pricing", by David Hirshleifer (2001)

### Identification and modeling of alpha factors

- The Barra Equity Risk Model Handbook
- Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk by Richard Grinold and Ronald Kahn
- Modern Investment Management: An Equilibrium Approach by Bob Litterman
- Quantitative Equity Portfolio Management: Modern Techniques and Applications by Edward Qian, Ronald Hua, and Eric Sorensen