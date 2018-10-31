# Chapter 05: Strategy Evaluation & Portfolio Management

This chapter covers:

- How to build and test a portfolio based on alpha factors using zipline
- How to measure portfolio risk and return
- How to evaluate portfolio performance using pyfolio
- How to manage portfolio weights using mean-variance optimization and alternatives
- How to use machine learning to optimize asset allocation in a portfolio context

The notebook content reflects this organization:

## 01 Trading with `zipline`

This directory contains a python file with the code required to simulate the trading decisions that build a portfolio based on the simple alpha factor from the last chapter using zipline.

## 02 Risk Metris with `pyfolio`

You'll find a jupyter notebook that illustrates how to extract the `pyfolio` input from the backtest conducted in the previous folder. It then proceeds to calcuate several performance metrics and tear sheets using `pyfolio`

## 03 (Multiple) Backtesting

This directory contains the implementation of the Deflated Sharpe Ratio by [Marcos Lopez de Prado](http://www.quantresearch.info/Software.htm).


## 04 Efficient Frontier

## 05 Kelly Rule


## References

- [Backtest Overfitting: An Interactive Example](http://datagrid.lbl.gov/backtest/)
- [Beat the Market: A Scientific Stock Market System](https://www.researchgate.net/publication/275756748_Beat_the_Market_A_Scientific_Stock_Market_System)
- [Markowitz PF Theory](https://www.math.ust.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf)

## Resources

- [Backtest overfitting simulator](http://datagrid.lbl.gov/backtest/)
- Marcos Lopez de Prado quantresearch [website](http://www.quantresearch.info/)
