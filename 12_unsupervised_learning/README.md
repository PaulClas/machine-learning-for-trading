# Chapter 12: Unsupervised Learning


## Content


## References

### Dimensionality Reduction

- [Dimension Reduction: A Guided Tour](https://www.microsoft.com/en-us/research/publication/dimension-reduction-a-guided-tour-2/), Chris J.C. Burges, Foundations and Trends in Machine Learning, January 2010

#### PCA

- [Mixtures of Probabilistic Principal Component Analysers](http://www.miketipping.com/papers/met-mppca.pdf), Michael E. Tipping and Christopher M. Bishop, Neural Computation 11(2), pp 443–482. MIT Press
- [Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions](http://users.cms.caltech.edu/~jtropp/papers/HMT11-Finding-Structure-SIREV.pdf), N. Halko†, P. G. Martinsson, J. A. Tropp, SIAM REVIEW, Vol. 53, No. 2, pp. 217–288
- [Relationship between SVD and PCA. How to use SVD to perform PCA?](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca), excellent technical CrossValidated StackExchange answer with visualization

#### ICA

- [Independent Component Analysis: Algorithms and Applications](https://www.sciencedirect.com/science/article/pii/S0893608000000265), Aapo Hyvärinen and Erkki Oja, Neural Networks, 2000
- [Independent Components Analysis](http://cs229.stanford.edu/notes/cs229-notes11.pdf), CS229 Lecture Notes, Andrew Ng

### Hierarchical Risk Parity

- [Building Diversified Portfolios that Outperform Out-of-Sample](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678), Lopez de Prado, Journal of Portfolio Management, 2015
- [Hierarchical Clustering Based Asset Allocation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840729), Raffinot 2016


pca = PCA()
projected_data  = pca.fit_transform(data)
projected_data.shape

C = pca.components_.T # columns = principal components
C