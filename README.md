# iSFun: Integrative Dimension Reduction Analysis for Multi-Source Data
## Introduction
The implement of integrative analysis methods based on a two-part penalization, which realizes dimension reduction analysis and mining the heterogeneity and association of multiple studies with compatible designs. The software package provides the integrative analysis methods including integrative sparse principal component analysis (Fang et al., 2018), integrative sparse partial least squares (Liang et al., 2021) and integrative sparse canonical correlation analysis, as well as corresponding individual analysis and meta-analysis versions.

***References:***
(1) Fang, K., Fan, X., Zhang, Q., and Ma, S. (2018). Integrative sparse principal component analysis. Journal of Multivariate Analysis, <doi:10.1016/j.jmva.2018.02.002>. 
(2) Liang, W., Ma, S., Zhang, Q., and Zhu, T. (2021). Integrative sparse partial least squares. Statistics in Medicine, <doi:10.1002/sim.8900>.
## Installation
***Requirements:*** <br />
matplotlib==3.3.4<br />
numpy==1.20.1<br />
pandas==1.2.4<br />
scikit_learn==0.24.1<br />
scipy==1.6.2<br />
seaborn==0.11.1<br />
setuptools==68.2.2<br />
```c
pip install iSFun
```
## Functions
### Menu
- [iscca](#iscca)
- [iscca_cv](#iscca_cv)
- 
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
-
#### iscca
Integrative sparse canonical correlation analysis.
##### Description
This function provides a penalty-based integrative sparse canonical correlation analysis method to handle the multiple datasets with high dimensions generated under similar protocols, which consists of two built-in penalty items for selecting the important variables for users to choose, and two contrasted penalty functions for eliminating the diffierence (magnitude or sign) between estimators within each group.
##### Usage
```c
iscca(x, y, L, mu1, mu2, mu3, mu4, eps=1e-04, pen1="homogeneity", 
          pen2="magnitude", scale_x=True, scale_y=True, maxstep=50, 
          submaxstep=10, trace=False, draw=False):
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
|x|list of data matrices, L datasets of explanatory variables.|
|y|list of data matrices, L datasets of dependent variables.|
L|numeric, number of datasets.|
mu1|numeric, sparsity penalty parameter for vector u.|
mu2|numeric, contrasted penalty parameter for vector u.|
mu3|numeric, sparsity penalty parameter for vector v.|
mu4|numeric, contrasted penalty parameter for vector v.|
eps|numeric, the threshold at which the algorithm terminates.|
pen1|character, "homogeneity" or "heterogeneity" type of the sparsity structure. If not specified, the default is homogeneity.|
pen2|character, "magnitude" or "sign" based contrasted penalty. If not specified, the default is magnitude.|
scale_x|character, "TRUE" or "FALSE", whether or not to scale the variables x. The default is TRUE.|
scale_y|character, "TRUE" or "FALSE", whether or not to scale the variables y. The default is TRUE.|
maxstep|numeric, maximum iteration steps. The default value is 50.|
submaxstep|numeric, maximum iteration steps in the sub-iterations. The default value is 10.|
trace|character, "TRUE" or "FALSE". If TRUE, prints out its screening results of variables.|
draw|character, "TRUE" or "FALSE". If TRUE, plot the convergence path of loadings and the heatmap of coefficient beta.|
##### Value
An 'iscca' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is TRUE, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- y: list of data matrices, L datasets of dependent variables with centered columns. If scale_y is TRUE, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- loading_x: the estimated canonical vector of variables x.
- loading_y: the estimated canonical vector of variables y.
- variable_x: the screening results of variables x.
- variable_y: the screening results of variables y.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
- meany: list of numeric vectors, column mean of the original datasets y.
- normy: list of numeric vectors, column standard deviation of the original datasets y.
##### See Also
See Also as [preview_cca](#preview_cca), [iscca_cv](#iscca_cv), [meta_scca](#meta_scca), scca[scca](#scca).
##### Examples
```c
from isfun_data import cca_data
x = cca_data()['x']
y = cca_data()['y']
L = len(x)
mu1 = mu3 = 0.4
mu2 = mu4 = 2.5

res_homo_m = iscca(x = x, y = y, L = L, mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4,
                   eps = 5e-2, maxstep = 50, submaxstep = 10, trace = True, draw = True)
res_homo_s = iscca(x=x, y=y, L=L, mu1=mu1, mu2=mu2, mu3=mu3, mu4=mu4,
                    eps=5e-2, pen1="homogeneity", pen2="sign", scale_x=True,
                    scale_y=True, maxstep=50, submaxstep=10, trace=False, draw=False)

mu1 = mu3 = 0.3
mu2 = mu4 = 2

res_hete_m = iscca(x = x, y = y, L = L, mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4,
                    eps = 5e-2, pen1 = "heterogeneity", pen2 = "magnitude", scale_x = True,
                    scale_y = True, maxstep = 50, submaxstep = 10, trace = False, draw = False)
res_hete_s = iscca(x = x, y = y, L = L, mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4,
                    eps = 5e-2, pen1 = "heterogeneity", pen2 = "sign", scale_x = True,
                    scale_y = True, maxstep = 50, submaxstep = 10, trace = False, draw = False)
```
#### iscca_cv

