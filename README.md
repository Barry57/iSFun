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
- [cca_data](#cca_data)
- [pca_data](#pca_data)
- [pls_data](#pls_data)
- [iscca](#iscca)
- [iscca_cv](#iscca_cv)
- [iscca_plot](#iscca_plot)
- [ispca](#ispca)
- [ispca_cv](#ispca_cv)
- [ispca_plot](#ispca_plot)
- [ispls](#ispls)
- [ispls_cv](#ispls_cv)
- [ispls_plot](#ispls_plot)
- [preview_cca](#preview_cca)
- [preview_pca](#preview_pca)
- [preview_pls](#preview_pls)
- [scca](#scca)
- [spca](#spca)
- [spls](#spls)
#### cca_data
*Example data for method iscca*
##### Description
Example data for users to apply the method iscca, iscca.cv, meta.scca or scca.
##### Format
A dict contains two lists 'x' and 'y'.
<br />
<br />
<br />

#### pca_data
*Example data for method ispca*
##### Description
Example data for users to apply the method ispca, ispca.cv, meta.spca or spca.
##### Format
A dict contains a list 'x'.
<br />
<br />
<br />

#### pls_data
*Example data for method ispls*
##### Description
Example data for users to apply the method ispls, ispls.cv, meta.spls or spls.
##### Format
A dict contains two lists 'x' and 'y'.
<br />
<br />
<br />

#### iscca
*Integrative sparse canonical correlation analysis*
##### Description
This function provides a penalty-based integrative sparse canonical correlation analysis method to handle the multiple datasets with high dimensions generated under similar protocols, which consists of two built-in penalty items for selecting the important variables for users to choose, and two contrasted penalty functions for eliminating the diffierence (magnitude or sign) between estimators within each group.
##### Usage
```c
iscca(x, y, L, mu1, mu2, mu3, mu4, eps=1e-04, pen1="homogeneity", pen2="magnitude",
      scale_x=True, scale_y=True, maxstep=50, submaxstep=10, trace=False, draw=False)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
y|list of data matrices, L datasets of dependent variables.
L|numeric, number of datasets.
mu1|numeric, sparsity penalty parameter for vector u.
mu2|numeric, contrasted penalty parameter for vector u.
mu3|numeric, sparsity penalty parameter for vector v.
mu4|numeric, contrasted penalty parameter for vector v.
eps|numeric, the threshold at which the algorithm terminates.
pen1|character, "homogeneity" or "heterogeneity" type of the sparsity structure. If not specified, the default is homogeneity.
pen2|character, "magnitude" or "sign" based contrasted penalty. If not specified, the default is magnitude.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
scale_y|character, "True" or "False", whether or not to scale the variables y. The default is True.
maxstep|numeric, maximum iteration steps. The default value is 50.
submaxstep|numeric, maximum iteration steps in the sub-iterations. The default value is 10.
trace|character, "True" or "False". If True, prints out its screening results of variables.
draw|character, "True" or "False". If True, plot the convergence path of loadings and the heatmap of coefficient beta.
##### Value
An 'iscca' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- y: list of data matrices, L datasets of dependent variables with centered columns. If scale_y is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- loading_x: the estimated canonical vector of variables x.
- loading_y: the estimated canonical vector of variables y.
- variable_x: the screening results of variables x.
- variable_y: the screening results of variables y.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
- meany: list of numeric vectors, column mean of the original datasets y.
- normy: list of numeric vectors, column standard deviation of the original datasets y.
##### See Also
See Also as [preview_cca](#preview_cca), [iscca_cv](#iscca_cv), [meta_scca](#meta_scca), [scca](#scca).
##### Examples
```c
from isfun_data import cca_data
from isfun import iscca
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
<br />
<br />

#### iscca_cv
*Cross-validation for iscca*
##### Description
Performs K-fold cross validation for the integrative sparse canonical correlation analysis over a grid of values for the regularization parameter mu1, mu2, mu3 and mu4.
##### Usage
```c
iscca_cv(x, y, L, mu1, mu2, mu3, mu4, K=5, eps=1e-04, pen1="homogeneity", pen2="magnitude",
         scale_x=True, scale_y=True, maxstep=50, submaxstep=10)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
y|list of data matrices, L datasets of dependent variables.
L|numeric, number of datasets.
K|numeric, number of cross-validation folds. Default is 5.
mu1|numeric, sparsity penalty parameter for vector u.
mu2|numeric, contrasted penalty parameter for vector u.
mu3|numeric, sparsity penalty parameter for vector v.
mu4|numeric, contrasted penalty parameter for vector v.
eps|numeric, the threshold at which the algorithm terminates.
pen1|character, "homogeneity" or "heterogeneity" type of the sparsity structure. If not specified, the default is homogeneity.
pen2|character, "magnitude" or "sign" based contrasted penalty. If not specified, the default is magnitude.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
scale_y|character, "True" or "False", whether or not to scale the variables y. The default is True.
maxstep|numeric, maximum iteration steps. The default value is 50.
submaxstep|numeric, maximum iteration steps in the sub-iterations. The default value is 10.
##### Value
An 'iscca_cv' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- y: list of data matrices, L datasets of dependent variables with centered columns. If scale_y is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- mu1: the sparsity penalty parameter selected from the feasible set of parameter mu1 provided by users.
- mu2: the contrasted penalty parameter selected from the feasible set of parameter mu2 provided by users.
- mu3: the sparsity penalty parameter selected from the feasible set of parameter mu3 provided by users.
- mu4: the contrasted penalty parameter selected from the feasible set of parameter mu4 provided by users.
- fold: The fold assignments for cross-validation for each observation.
- loading_x: the estimated canonical vector of variables x with selected tuning parameters.
- loading_y: the estimated canonical vector of variables y with selected tuning parameters.
- variable_x: the screening results of variables x.
- variable_y: the screening results of variables y.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
- meany: list of numeric vectors, column mean of the original datasets y.
- normy: list of numeric vectors, column standard deviation of the original datasets y.
##### See Also
See Also as [iscca](#iscca).
##### Examples
```c
from isfun_data import cca_data
from isfun import iscca_cv
x = cca_data()['x']
y = cca_data()['y']
L = len(x)

mu1 = [0.2, 0.4]
mu3 = 0.4
mu2 = mu4 = 2.5
res_homo_m = iscca_cv(x = x, y = y, L = L, mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4, K = 5,
                      eps = 1e-2, pen1="homogeneity", pen2="magnitude", scale_x=True,
                      scale_y = True, maxstep = 50, submaxstep = 10)
res_homo_s = iscca_cv(x = x, y = y, L = L, mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4, K = 5, 
                      eps = 1e-2, pen1 = "homogeneity", pen2 = "sign", scale_x = True,
                      scale_y = True, maxstep = 50, submaxstep = 10)

mu1 = mu3 = [0.1, 0.3]
mu2 = mu4 = 2
res_hete_m = iscca_cv(x = x, y = y, L = L, mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4, K = 5,
                      eps = 1e-2, pen1 = "heterogeneity", pen2 = "magnitude", scale_x = True,
                      scale_y = True, maxstep = 50, submaxstep = 10)
res_hete_s = iscca_cv(x = x, y = y, L = L, mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4, K = 5,
                      eps = 1e-2, pen1 = "heterogeneity", pen2 = "sign", scale_x = True,
                      scale_y = True, maxstep = 50, submaxstep = 10)
```
<br />
<br />

#### iscca_plot
*Plot the results of iscca*
##### Description
Plot the convergence path graph in the integrative sparse canonical correlation analysis method or show the the first pair of canonical vectors.
##### Usage
```c
iscca_plot(x, type)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of "iscca", which is the result of command "iscca".
type|character, "path" or "loading" type, if "path", plot the the convergence path graph of vector u and v in the integrative sparse canonical correlation analysis method, if "loading", show the the first pair of canonical vectors.
##### Details
See details in [iscca](#iscca).
##### Value
The convergence path graph or the scatter diagrams of the first pair of canonical vectors.
##### Examples
```c
from isfun_data import cca_data
from isfun import iscca
from isfun import iscca_plot
x = cca_data()['x']
y = cca_data()['y']
L = len(x)

mu1 = mu3 = 0.4
mu2 = mu4 = 2.5
res_homo_m = iscca(x=x, y=y, L=L, mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4,
                   eps = 5e-2, maxstep = 100, trace = False, draw = False)
iscca_plot(x = res_homo_m, type = "path")
iscca_plot(x = res_homo_m, type = "loading")
```
<br />
<br />

#### ispca
*Integrative sparse principal component analysis*
##### Description
This function provides a penalty-based integrative sparse principal component analysis method to obtain the direction of first principal component of the multiple datasets with high dimensions generated under similar protocols, which consists of two built-in penalty items for selecting the important variables for users to choose, and two contrasted penalty functions for eliminating the diffierence (magnitude or sign) between estimators within each group.
##### Usage
```c
ispca(x, L, mu1, mu2, eps = 1e-04, pen1 = "homogeneity", pen2 = "magnitude",
      scale_x = True, maxstep = 50, submaxstep = 10, trace = False, draw = False)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
L|numeric, number of datasets.
mu1|numeric, sparsity penalty parameter.
mu2|numeric, contrasted penalty parameter.
eps|numeric, the threshold at which the algorithm terminates.
pen1|character, "homogeneity" or "heterogeneity" type of the sparsity structure. If not specified, the default is homogeneity.
pen2|character, "magnitude" or "sign" based contrasted penalty. If not specified, the default is magnitude.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
maxstep|numeric, maximum iteration steps. The default value is 50.
submaxstep|numeric, maximum iteration steps in the sub-iterations. The default value is 10.
trace|character, "True" or "False". If True, prints out its screening results of variables.
draw|character, "True" or "False". If True, plot the convergence path of loadings and the heatmap of coefficient beta.
##### Value
An 'ispca' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- eigenvalue: the estimated first eigenvalue.
- eigenvector: the estimated first eigenvector.
- component: the estimated first component.
- variable: the screening results of variables.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
##### References
Fang K, Fan X, Zhang Q, et al. Integrative sparse principal component analysis[J]. Journal of Multivariate Analysis, 2018, 166: 1-16.
##### See Also
See Also as [preview_pca](#preview_pca), [ispca_cv](#ispca_cv), [meta_spca](#meta_spca), [spca](#spca).
##### Examples
```c
from isfun_data import pca_data
from isfun import ispca
x = pca_data()['x']
L = len(x)

res_homo_m = ispca(x = x, L = L, mu1 = 0.5, mu2 = 0.002, trace = True, draw = True)
res_homo_s = ispca(x = x, L = L, mu1 = 0.5, mu2 = 0.002,
                   pen1 = "homogeneity", pen2 = "sign", scale_x = True,
                   maxstep = 50, submaxstep = 10, trace = False, draw = False)

res_hete_m = ispca(x = x, L = L, mu1 = 0.1, mu2 = 0.05,
                   pen1 = "heterogeneity", pen2 = "magnitude", scale_x = True,
                   maxstep = 50, submaxstep = 10, trace = False, draw = False)
res_hete_s = ispca(x = x, L = L, mu1 = 0.1, mu2 = 0.05,
                   pen1 = "heterogeneity", pen2 = "sign", scale_x = True,
                   maxstep = 50, submaxstep = 10, trace = False, draw = False)
```
<br />
<br />

#### ispca_cv
*Cross-validation for ispca*
##### Description
Performs K-fold cross validation for the integrative sparse principal component analysis over a grid of values for the regularization parameter mu1 and mu2.
##### Usage
```c
ispca_cv(x, L, mu1, mu2, K = 5, eps = 1e-04, pen1 = "homogeneity", 
         pen2 = "magnitude", scale_x = True, maxstep = 50, submaxstep = 10)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
L|numeric, number of datasets.
K|numeric, number of cross-validation folds. Default is 5.
mu1|numeric, sparsity penalty parameter.
mu2|numeric, contrasted penalty parameter.
eps|numeric, the threshold at which the algorithm terminates.
pen1|character, "homogeneity" or "heterogeneity" type of the sparsity structure. If not specified, the default is homogeneity.
pen2|character, "magnitude" or "sign" based contrasted penalty. If not specified, the default is magnitude.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
maxstep|numeric, maximum iteration steps. The default value is 50.
submaxstep|numeric, maximum iteration steps in the sub-iterations. The default value is 10.
##### Value
An 'ispca_cv' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- mu1: the sparsity penalty parameter selected from the feasible set of parameter mu1 provided by users.
- mu2: the contrasted penalty parameter selected from the feasible set of parameter mu2 provided by users.
- fold: The fold assignments for cross-validation for each observation.
- eigenvalue: the estimated first eigenvalue with selected tuning parameters mu1 and mu2.
- eigenvector: the estimated first eigenvector with selected tuning parameters mu1 and mu2.
- component: the estimated first component with selected tuning parameters mu1 and mu2.
- variable: the screening results of variables.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
##### References
Fang K, Fan X, Zhang Q, et al. Integrative sparse principal component analysis[J]. Journal of Multivariate Analysis, 2018, 166: 1-16.
##### See Also
See Also as [ispca](#ispca).
##### Examples
```c
from isfun_data import pca_data
from isfun import ispca_cv
x = pca_data()['x']
L = len(x)

mu1 = [0.3, 0.5]
mu2 = 0.002
res_homo_m = ispca_cv(x = x, L = L, mu1 = mu1, mu2 = mu2, pen1 = "homogeneity", K = 5,
                      pen2 = "magnitude", scale_x = True, maxstep = 50, submaxstep = 10)
res_homo_s = ispca_cv(x = x, L = L, mu1 = mu1, mu2 = mu2, pen1 = "homogeneity", K = 5,
                      pen2 = "sign", scale_x = True, maxstep = 50, submaxstep = 10)

mu1 = [0.1, 0.15]
mu2 = 0.05
res_hete_m = ispca_cv(x = x, L = L, mu1 = mu1, mu2 = mu2, pen1 = "heterogeneity", K = 5,
                       pen2 = "magnitude", scale_x = True, maxstep = 50, submaxstep = 10)
res_hete_s = ispca_cv(x = x, L = L, mu1 = mu1, mu2 = mu2, pen1 = "heterogeneity", K = 5,
                       pen2 = "sign", scale_x = True, maxstep = 50, submaxstep = 10)
```
<br />
<br />

#### ispca_plot
*Plot the results of ispca*
##### Description
Plot the convergence path graph or estimated value of the first eigenvector u in the integrative sparse principal component analysis method.
##### Usage
```c
ispca_plot(x, type)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of "ispca", which is the result of command "ispca".
type|character, "path" or "loading" type, if "path", plot the the convergence path graph of vector u and v in the integrative sparse canonical correlation analysis method, if "loading", show the the first pair of canonical vectors.
##### Details
See details in [ispca](#ispca).
##### Value
The convergence path graph or the scatter diagrams of the first eigenvector u.
##### Examples
```c
from isfun_data import pca_data
from isfun import ispca
from isfun import ispca_plot
x = pca_data()['x']
L = len(x)

res_homo_m = ispca(x=x, L=L, mu1 = 0.5, mu2 = 0.002, trace = False, draw = False)
ispca_plot(x = res_homo_m, type = "path")
ispca_plot(x = res_homo_m, type = "loading")
```
<br />
<br />

#### ispls
*Integrative sparse partial least squares*
##### Description
This function provides a penalty-based integrative sparse partial least squares method to handle the multiple datasets with high dimensions generated under similar protocols, which consists of two built-in penalty items for selecting the important variables for users to choose, and two contrasted penalty functions for eliminating the diffierence (magnitude or sign) between estimators within each group.
##### Usage
```c
ispls(x, y, L, mu1, mu2, eps=1e-04, kappa=0.05, pen1="homogeneity", pen2="magnitude",
      scale_x=True, scale_y=True, maxstep=50, submaxstep=10, trace=False, draw=False)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
y|list of data matrices, L datasets of dependent variables.
L|numeric, number of datasets.
mu1|numeric, sparsity penalty parameter
mu2|numeric, contrasted penalty parameter.
eps|numeric, the threshold at which the algorithm terminates.
kappa|numeric, 0 < kappa < 0.5 and the parameter reduces the effect of the concave part of objective function.
pen1|character, "homogeneity" or "heterogeneity" type of the sparsity structure. If not specified, the default is homogeneity.
pen2|character, "magnitude" or "sign" based contrasted penalty. If not specified, the default is magnitude.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
scale_y|character, "True" or "False", whether or not to scale the variables y. The default is True.
maxstep|numeric, maximum iteration steps. The default value is 50.
submaxstep|numeric, maximum iteration steps in the sub-iterations. The default value is 10.
trace|character, "True" or "False". If True, prints out its screening results of variables.
draw|character, "True" or "False". If True, plot the convergence path of loadings and the heatmap of coefficient beta.
##### Value
An 'ispls' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- y: list of data matrices, L datasets of dependent variables with centered columns. If scale_y is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- betahat: the estimated regression coefficients.
- loading: the estimated first direction vector.
- variable: the screening results of variables x.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
- meany: list of numeric vectors, column mean of the original datasets y.
- normy: list of numeric vectors, column standard deviation of the original datasets y.
##### References
Liang W, Ma S, Zhang Q, et al. Integrative sparse partial least squares[J]. Statistics in Medicine, 2021, 40(9): 2239-2256.
##### See Also
See Also as [preview_pls](#preview_pls), [ispls_cv](#ispls_cv), [meta_spls](#meta_spls), [spls](#spls).
##### Examples
```c
from isfun_data import pls_data
from isfun import ispls
x = pls_data()['x']
y = pls_data()['y']
L = len(x)

res_homo_m = ispls(x = x, y = y, L = L, mu1 = 0.05, mu2 = 0.25,
                   eps = 5e-2, trace = True, draw = True)
res_homo_s = ispls(x = x, y = y, L = L, mu1 = 0.05, mu2 = 0.25,
                   eps = 5e-2, kappa = 0.05, pen1 = "homogeneity",
                   pen2 = "sign", scale_x = True, scale_y = True,
                   maxstep = 50, submaxstep = 10, trace = False, draw = False)

res_hete_m = ispls(x = x, y = y, L = L, mu1 = 0.05, mu2 = 0.25,
                   eps = 5e-2, kappa = 0.05, pen1 = "heterogeneity",
                   pen2 = "magnitude", scale_x = True, scale_y = True,
                   maxstep = 50, submaxstep = 10, trace = False, draw = False)
res_hete_s = ispls(x = x, y = y, L = L, mu1 = 0.05, mu2 = 0.25,
                   eps = 5e-2, kappa = 0.05, pen1 = "heterogeneity",
                   pen2 = "sign", scale_x = True, scale_y = True,
                   maxstep = 50, submaxstep = 10, trace = False, draw = False)
```
<br />
<br />

#### ispls_cv
*Cross-validation for ispls*
##### Description
Performs K-fold cross validation for the integrative sparse partial least squares over a grid of values for the regularization parameter mu1 and mu2.
##### Usage
```c
ispls_cv(x, y, L, K, mu1, mu2, eps=1e-04, kappa=0.05, pen1="homogeneity", 
         pen2="magnitude", scale_x=True, scale_y=True, maxstep=50, submaxstep=10)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
y|list of data matrices, L datasets of dependent variables.
L|numeric, number of datasets.
K|numeric, number of cross-validation folds. Default is 5.
mu1|numeric, sparsity penalty parameter
mu2|numeric, contrasted penalty parameter.
eps|numeric, the threshold at which the algorithm terminates.
kappa|numeric, 0 < kappa < 0.5 and the parameter reduces the effect of the concave part of objective function.
pen1|character, "homogeneity" or "heterogeneity" type of the sparsity structure. If not specified, the default is homogeneity.
pen2|character, "magnitude" or "sign" based contrasted penalty. If not specified, the default is magnitude.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
scale_y|character, "True" or "False", whether or not to scale the variables y. The default is True.
maxstep|numeric, maximum iteration steps. The default value is 50.
submaxstep|numeric, maximum iteration steps in the sub-iterations. The default value is 10.
##### Value
An 'ispls_cv' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- y: list of data matrices, L datasets of dependent variables with centered columns. If scale_y is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- mu1: the sparsity penalty parameter selected from the feasible set of parameter mu1 provided
by users.
- mu2: the contrasted penalty parameter selected from the feasible set of parameter mu2 provided by users.
- fold: The fold assignments for cross-validation for each observation.
- betahat: the estimated regression coefficients with selected tuning parameters mu1 and mu2.
- loading: the estimated first direction vector with selected tuning parameters mu1 and mu2.
- variable: the screening results of variables x.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
- meany: list of numeric vectors, column mean of the original datasets y.
- normy: list of numeric vectors, column standard deviation of the original datasets y.
##### References
Liang W, Ma S, Zhang Q, et al. Integrative sparse partial least squares[J]. Statistics in Medicine, 2021, 40(9): 2239-2256.
##### See Also
See Also as [ispls](#ispls).
##### Examples
```c
from isfun_data import pls_data
from isfun import ispls_cv
x = pls_data()['x']
y = pls_data()['y']
L = len(x)

mu1 = [0.04, 0.05]
mu2 = 0.25
res_homo_m = ispls_cv(x = x, y = y, L = L, K = 5, mu1 = mu1, mu2 = mu2, eps = 1e-2,
                      kappa = 0.05, pen1 = "homogeneity", pen2 = "magnitude",
                      scale_x = True, scale_y = True, maxstep = 50, submaxstep = 10)
res_homo_s = ispls_cv(x = x, y = y, L = L, K = 5, mu1 = mu1, mu2 = mu2, eps = 1e-2,
                      kappa = 0.05, pen1 = "homogeneity", pen2 = "sign",
                      scale_x = True, scale_y = True, maxstep = 50, submaxstep = 10)

res_hete_m = ispls_cv(x = x, y = y, L = L, K = 5, mu1 = mu1, mu2 = mu2, eps = 1e-2,
                      kappa = 0.05, pen1 = "heterogeneity", pen2 = "magnitude",
                      scale_x = True, scale_y = True, maxstep = 50, submaxstep = 10)
res_hete_s = ispls_cv(x = x, y = y, L = L, K = 5, mu1 = mu1, mu2 = mu2, eps = 1e-2,
                      kappa = 0.05, pen1 = "heterogeneity", pen2 = "sign",
                      scale_x = True, scale_y = True, maxstep = 50, submaxstep = 10)
```
<br />
<br />

#### ispls_plot
*Plot the results of ispls*
##### Description
Plot the convergence path graph of the first direction vector w in the integrative sparse partial least squares model or show the regression coefficients.
##### Usage
```c
ispls_plot(x, type)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of "ispls", which is the result of command "ispls".
type|character, "path", "loading" or "heatmap" type, if "path", plot the the convergence path graph of vector w in the integrative sparse partial least squares model,if "loading", plot the the first direction vectors, if "heatmap", show the heatmap of regression coefficients among different datasets.
##### Details
See details in [ispls](#ispls).
##### Value
Show the convergence path graph of the first direction vector w or the regression coefficients.
##### Examples
```c
from isfun_data import pls_data
from isfun import ispls
from isfun import ispls_plot
x = pls_data()['x']
y = pls_data()['y']
L = len(x)

res_homo_m = ispls(x = x, y = y, L = L, mu1 = 0.05, mu2 = 0.25,
                    eps = 5e-2, trace = False, draw = False)
ispls_plot(x = res_homo_m, type = "path")
ispls_plot(x = res_homo_m, type = "loading")
ispls_plot(x = res_homo_m, type = "heatmap")
```
<br />
<br />

#### meta_scca
*Meta-analytic sparse canonical correlation analysis method in integrative study*
##### Description
This function provides penalty-based sparse canonical correlation meta-analytic method to handle the multiple datasets with high dimensions generated under similar protocols, which is based on the principle of maximizing the summary statistics S.
##### Usage
```c
meta_scca(x, y, L, mu1, mu2, eps=1e-04, scale_x=True, scale_y=True, maxstep=50, trace=False)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
y|list of data matrices, L datasets of dependent variables.
L|numeric, number of datasets.
mu1|numeric, sparsity penalty parameter for vector u.
mu2|numeric, sparsity penalty parameter for vector v.
eps|numeric, the threshold at which the algorithm terminates.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
scale_y|character, "True" or "False", whether or not to scale the variables y. The default is True.
maxstep|numeric, maximum iteration steps. The default value is 50.
trace|character, "True" or "False". If True, prints out its screening results of variables.
##### Value
An 'meta_scca' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- y: list of data matrices, L datasets of dependent variables with centered columns. If scale_y is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- loading_x: the estimated canonical vector of variables x.
- loading_y: the estimated canonical vector of variables y.
- variable_x: the screening results of variables x.
- variable_y: the screening results of variables y.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
- meany: list of numeric vectors, column mean of the original datasets y.
- normy: list of numeric vectors, column standard deviation of the original datasets y.
##### References
Cichonska A, Rousu J, Marttinen P, et al. metaCCA: summary statistics-based multivariate meta-analysis of genome-wide association studies using canonical correlation analysis[J]. Bioinformatics, 2016, 32(13): 1981-1989.
##### See Also
See Also as [iscca](#iscca), [scca](#scca).
##### Examples
```c
from isfun_data import cca_data
from isfun import meta_scca
x = cca_data()['x']
y = cca_data()['y']
L = len(x)

mu1 = 0.08
mu2 = 0.08
res = meta_scca(x = x, y = y, L = L, mu1 = mu1, mu2 = mu2, trace = True)
```
<br />
<br />

#### meta_spca
*Meta-analytic sparse principal component analysis method in integrative study*
##### Description
This function provides penalty-based sparse principal component meta-analytic method to handle the multiple datasets with high dimensions generated under similar protocols, which is based on the principle of maximizing the summary statistics S.
##### Usage
```c
meta_spca(x, L, mu1, eps=1e-04, scale_x=True, maxstep=50, trace=False)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
L|numeric, number of datasets.
mu1|numeric, sparsity penalty parameter.
eps|numeric, the threshold at which the algorithm terminates.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
maxstep|numeric, maximum iteration steps. The default value is 50.
trace|character, "True" or "False". If True, prints out its screening results of variables.
##### Value
An 'meta_spca' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- eigenvalue: the estimated first eigenvalue.
- eigenvector: the estimated first eigenvector.
- component: the estimated first component.
- variable: the screening results of variables.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
##### References
Kim S H, Kang D, Huo Z, et al. Meta-analytic principal component analysis in integrative omics application[J]. Bioinformatics, 2018, 34(8): 1321-1328.
##### See Also
See Also as [ispca](#ispca), [spca](#spca).
##### Examples
```c
from isfun_data import pca_data
from isfun import meta_spca
x = pca_data()['x']
L = len(x)

res = meta_spca(x = x, L = L, mu1 = 0.5, trace = True)
```
<br />
<br />

#### meta_spls
*Meta-analytic sparse partial least squares method in integrative study*
##### Description
This function provides penalty-based sparse canonical correlation meta-analytic method to handle the multiple datasets with high dimensions generated under similar protocols, which is based on the principle of maximizing the summary statistics.
##### Usage
```c
meta_spls(x, y, L, mu1, eps=1e-04, kappa=0.05, scale_x=True, scale_y=True, maxstep=50, trace=False)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
y|list of data matrices, L datasets of dependent variables.
L|numeric, number of datasets.
mu1|numeric, sparsity penalty parameter
eps|numeric, the threshold at which the algorithm terminates.
kappa|numeric, 0 < kappa < 0.5 and the parameter reduces the effect of the concave part of objective function.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
scale_y|character, "True" or "False", whether or not to scale the variables y. The default is True.
maxstep|numeric, maximum iteration steps. The default value is 50.
trace|character, "True" or "False". If True, prints out its screening results of variables.
##### Value
An 'meta_spls' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- y: list of data matrices, L datasets of dependent variables with centered columns. If scale_y is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- betahat: the estimated regression coefficients.
- loading: the estimated first direction vector.
- variable: the screening results of variables x.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
- meany: list of numeric vectors, column mean of the original datasets y.
- normy: list of numeric vectors, column standard deviation of the original datasets y.
##### See Also
See Also as [ispls](#ispls), [spls](#spls).
##### Examples
```c
from isfun_data import pls_data
from isfun import meta_spls
x = pls_data()['x']
y = pls_data()['y']
L = len(x)

res = meta_spls(x = x, y = y, L = L, mu1 = 0.03, trace = True)
```
<br />
<br />

#### preview_cca
*Statistical description before using function iscca*
##### Description
The function describes the basic statistical information of the data, including sample mean, sample variance of X and Y, and the first pair of canonical vectors.
##### Usage
```c
preview_cca(x, y, L, scale_x=True, scale_y=True)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
y|list of data matrices, L datasets of dependent variables.
L|numeric, number of datasets.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
scale_y|character, "True" or "False", whether or not to scale the variables y. The default is True.
##### Value
An 'preview_cca' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- y: list of data matrices, L datasets of dependent variables with centered columns. If scale_y is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- loading_x: the estimated canonical vector of variables x.
- loading_y: the estimated canonical vector of variables y.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
- meany: list of numeric vectors, column mean of the original datasets y.
- normy: list of numeric vectors, column standard deviation of the original datasets y.
##### See Also
See Also as [iscca](#iscca).
##### Examples
```c
from isfun_data import cca_data
from isfun import preview_cca
x = cca_data()['x']
y = cca_data()['y']
L = len(x)

prev_cca = preview_cca(x = x, y = y, L = L, scale_x = True, scale_y = True)
```
<br />
<br />

#### preview_pca
*Statistical description before using function ispca*
##### Description
The function describes the basic statistical information of the data, including sample mean, sample co-variance of X and Y, the first eigenvector, eigenvalue and principal component, etc.
##### Usage
```c
preview_pca(x, L, scale_x=True)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
L|numeric, number of datasets.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
##### Value
An 'preview_pca' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- eigenvalue: the estimated first eigenvalue.
- eigenvector: the estimated first eigenvector.
- component: the estimated first component.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
##### See Also
See Also as [ispca](#ispca).
##### Examples
```c
from isfun_data import pca_data
from isfun import preview_pca
x = pca_data()['x']
L = len(x)

prev_pca = preview_pca(x = x, L = L, scale_x = True)
```
<br />
<br />

#### preview_pls
*Statistical description before using function ispls*
##### Description
The function describes the basic statistical information of the data, including sample mean, sample variance of X and Y, the first direction of partial least squares method, etc.
##### Usage
```c
preview_pls(x, y, L, scale_x=True, scale_y=True)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
y|list of data matrices, L datasets of dependent variables.
L|numeric, number of datasets.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
scale_y|character, "True" or "False", whether or not to scale the variables y. The default is True.
##### Value
An 'preview_pls' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- y: list of data matrices, L datasets of dependent variables with centered columns. If scale_y is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- loading: the estimated first direction vector.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
- meany: list of numeric vectors, column mean of the original datasets y.
- normy: list of numeric vectors, column standard deviation of the original datasets y.
##### See Also
See Also as [ispls](#ispls).
##### Examples
```c
from isfun_data import pls_data
from isfun import preview_pls
x = pls_data()['x']
y = pls_data()['y']
L = len(x)

prev_pls = preview_pls(x = x, y = y, L = L, scale_x = True, scale_y = True)
```
<br />
<br />

#### scca
*Sparse canonical correlation analysis*
##### Description
This function provides penalty-based sparse canonical correlation analysis to get the first pair of canonical vectors.
##### Usage
```c
scca(x, y, mu1, mu2, eps=1e-04, scale_x=True, scale_y=True, maxstep=50, trace=False)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
y|list of data matrices, L datasets of dependent variables.
mu1|numeric, sparsity penalty parameter for vector u.
mu2|numeric, sparsity penalty parameter for vector v.
eps|numeric, the threshold at which the algorithm terminates.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
scale_y|character, "True" or "False", whether or not to scale the variables y. The default is True.
maxstep|numeric, maximum iteration steps. The default value is 50.
trace|character, "True" or "False". If True, prints out its screening results of variables.
##### Value
An 'scca' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- y: list of data matrices, L datasets of dependent variables with centered columns. If scale_y is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- loading_x: the estimated canonical vector of variables x.
- loading_y: the estimated canonical vector of variables y.
- variable_x: the screening results of variables x.
- variable_y: the screening results of variables y.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
- meany: list of numeric vectors, column mean of the original datasets y.
- normy: list of numeric vectors, column standard deviation of the original datasets y.
##### See Also
See Also as [iscca](#iscca), [meta_scca](#meta_scca).
##### Examples
```c
import numpy as np
from isfun_data import cca_data
from isfun import scca
x_scca = np.vstack(cca_data()['x'])
y_scca = np.vstack(cca_data()['y'])

res_scca = scca(x = x_scca, y = y_scca, mu1 = 0.1, mu2 = 0.1, eps = 1e-3,
                scale_x = True, scale_y = True, maxstep = 50, trace = False)
```
<br />
<br />

#### spca
*Sparse principal component analysis*
##### Description
This function provides penalty-based integrative sparse principal component analysis to obtain the direction of first principal component of a given dataset with high dimensions.
##### Usage
```c
spca(x, mu1, eps=1e-04, scale_x=True, maxstep=50, trace=False)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
mu1|numeric, sparsity penalty parameter.
eps|numeric, the threshold at which the algorithm terminates.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
maxstep|numeric, maximum iteration steps. The default value is 50.
trace|character, "True" or "False". If True, prints out its screening results of variables.
##### Value
An 'spca' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- eigenvalue: the estimated first eigenvalue.
- eigenvector: the estimated first eigenvector.
- component: the estimated first component.
- variable: the screening results of variables.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
##### See Also
See Also as [ispca](#ispca), [meta_spca](#meta_spca).
##### Examples
```c
import numpy as np
from isfun_data import pca_data
from isfun import spca
x_spca = np.vstack(pca_data()['x'])

res_spca = spca(x = x_spca, mu1 = 0.08, eps = 1e-3, scale_x = True,
                maxstep = 50, trace = True)
```
<br />
<br />

#### spls
*Sparse partial least squares*
##### Description
This function provides penalty-based sparse partial least squares analysis for single dataset with high dimensions., which aims to have the direction of the first loading.
##### Usage
```c
spls(x, y, mu1, eps=1e-04, kappa=0.05, scale_x=True, scale_y=True, maxstep=50, trace=False):
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
x|list of data matrices, L datasets of explanatory variables.
y|list of data matrices, L datasets of dependent variables.
mu1|numeric, sparsity penalty parameter
eps|numeric, the threshold at which the algorithm terminates.
kappa|numeric, 0 < kappa < 0.5 and the parameter reduces the effect of the concave part of objective function.
scale_x|character, "True" or "False", whether or not to scale the variables x. The default is True.
scale_y|character, "True" or "False", whether or not to scale the variables y. The default is True.
maxstep|numeric, maximum iteration steps. The default value is 50.
trace|character, "True" or "False". If True, prints out its screening results of variables.
##### Value
An 'spls' object that contains the list of the following items.
- x: list of data matrices, L datasets of explanatory variables with centered columns. If scale_x is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- y: list of data matrices, L datasets of dependent variables with centered columns. If scale_y is True, the columns of L datasets are standardized to have mean 0 and standard deviation 1.
- betahat: the estimated regression coefficients.
- loading: the estimated first direction vector.
- variable: the screening results of variables x.
- meanx: list of numeric vectors, column mean of the original datasets x.
- normx: list of numeric vectors, column standard deviation of the original datasets x.
- meany: list of numeric vectors, column mean of the original datasets y.
- normy: list of numeric vectors, column standard deviation of the original datasets y.
##### See Also
See Also as [ispls](#ispls), [meta_spls](#meta_spls).
##### Examples
```c
import numpy as np
from isfun_data import pls_data
from isfun import spls
x_spls = np.vstack(pls_data()['x'])
y_spls = np.vstack(pls_data()['y'])

res_spls = spls(x = x_spls, y = y_spls, mu1 = 0.05, eps = 1e-3, kappa = 0.05,
                scale_x = True, scale_y = True, maxstep = 50, trace = True)
```
