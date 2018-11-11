# Chapter 02: Market & Fundamental Data


## Market Data

### 01_NASDAQ_TotalView-ITCH_Order_Book

This directory contains the code to download NASDAQ ITCH TotalView sample data, parse the data and reconstruct the order book.

### Resources

#### FIX implementations

 - Python: [Simplefix](https://github.com/da4089/simplefix)
 - C++ version: [quickfixengine](http://www.quickfixengine.org/)
 - Interactive Brokers [interface](https://www.interactivebrokers.com/en/index.php?f=4988)

#### ITCH Protocol

- The ITCH [Specifications](http://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHspecification.pdf)
- [Sample Files](ftp://emi.nasdaq.com/ITCH/)

#### Other protocols

 - Native exchange protocols [around the world](https://en.wikipedia.org/wiki/List_of_electronic_trading_protocols_

## Fundamental Data

### 01_NASDAQ_TotalView-ITCH_Order_Book

This folder contains the notebooks to
- download NASDAQ Total View sample tick data,
- parse the messages from the binary source data
- reconstruct the order book for a given stock
- visualize order flow data
- normalize tick data


### 02_EDGAR

This folder contains the code to download and parse EDGAR data in XBRL format.

## Resources

### 03_Data Providers

This folder contains examples to use various data providers.

### Python Libraries

#### HDF Format

- [Pandas HDF5](http://pandas.pydata.org/pandas-docs/version/0.22/io.html#hdf5-pytables)
- [HDF Support Portal](http://portal.hdfgroup.org/display/support)
- [PyTables](https://www.pytables.org/)

#### Parquet Format

- [Apache Parquet](https://parquet.apache.org/)
- [PyArrow: Parquet for Python](https://arrow.apache.org/docs/python/parquet.html)
- [Development update: High speed Apache Parquet in Python with Apache Arrow](http://wesmckinney.com/blog/python-parquet-update/)


#### Quantopian

- [Binary Data services: `struct`](https://docs.python.org/3/library/struct.html)


#### Quandl

#### pandas_datareader


## References

- [Trading and Exchanges - Market Microstructure for Practitioners](https://global.oup.com/ushe/product/trading-and-exchanges-9780195144703?cc=us&lang=en&), Larry Harris, Oxford University Press, 2002
- [World Federation of Exchanges](https://www.world-exchanges.org/our-work/statistics)
- [FIX Trading Standards](https://www.fixtrading.org/standards/)

## Data Sources

- [Compilation of macro resources by the Yale Law School](https://library.law.yale.edu/news/75-sources-economic-data-statistics-reports-and-commentary)
- [Capital IQ](www.capitaliq.com)
- [Compustat](www.compustat.com)
- [MSCI Barra](www.mscibarra.com)
- [Northfield Information Services](www.northinfo.com)
- [Quantitative Services Group](www.qsg.com)