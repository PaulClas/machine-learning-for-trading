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

- [The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), Bailey, David and Lopez de Prado, Marcos, Journal of Portfolio Management, 2013
- [Backtest Overfitting: An Interactive Example](http://datagrid.lbl.gov/backtest/)
- [Backtesting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2606462), Lopez de Prado, Marcos, 2015
- [Secretary Problem (Optimal Stopping)](https://www.geeksforgeeks.org/secretary-problem-optimal-stopping-problem/)
- [Optimal Stopping and Applications](https://www.math.ucla.edu/~tom/Stopping/Contents.html), Ferguson, Math Department, UCLA
- [Advances in Machine Learning Lectures 4/10 - Backtesting I](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257420), Marcos Lopez de Prado, 2018
- [Advances in Machine Learning Lectures 5/10 - Backtesting II](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257497), Marcos Lopez de Prado, 2018


## 04 Efficient Frontier

## 05 Kelly Rule


## References

- [Beat the Market: A Scientific Stock Market System](https://www.researchgate.net/publication/275756748_Beat_the_Market_A_Scientific_Stock_Market_System)
- [Markowitz PF Theory](https://www.math.ust.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf)

## Resources

- [Backtest overfitting simulator](http://datagrid.lbl.gov/backtest/)
- Marcos Lopez de Prado quantresearch [website](http://www.quantresearch.info/)
