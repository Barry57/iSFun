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
#### iscca
Integrative sparse canonical correlation analysis
##### Description
This function provides a penalty-based integrative sparse canonical correlation analysis method to handle the multiple datasets with high dimensions generated under similar protocols, which consists of two built-in penalty items for selecting the important variables for users to choose, and two contrasted penalty functions for eliminating the diffierence (magnitude or sign) between estimators within each group.
##### Usage
iscca(x, y, L, mu1, mu2, mu3, mu4, eps = 1e-04, pen1 = "homogeneity",
pen2 = "magnitude", scale.x = TRUE, scale.y = TRUE, maxstep = 50,
submaxstep = 10, trace = FALSE, draw = FALSE)
#####
