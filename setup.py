from setuptools import setup, find_packages

setup(
    name='iSFun',
    version='1.1.0',
    description='''
    The implement of integrative analysis methods based on a two-part penalization, 
    which realizes dimension reduction analysis and mining the heterogeneity and association of multiple studies with compatible designs. 
    The software package provides the integrative analysis methods including integrative sparse principal component analysis (Fang et al., 2018), 
    integrative sparse partial least squares (Liang et al., 2021) and integrative sparse canonical correlation analysis, 
    as well as corresponding individual analysis and meta-analysis versions. 
    References: (1) Fang, K., Fan, X., Zhang, Q., and Ma, S. (2018). Integrative sparse principal component analysis. Journal of Multivariate Analysis, <doi:10.1016/j.jmva.2018.02.002>. 
    (2) Liang, W., Ma, S., Zhang, Q., and Zhu, T. (2021). Integrative sparse partial least squares. Statistics in Medicine, <doi:10.1002/sim.8900>.',
    ''',
    author='Kuangnan Fang, Rui Ren, Qingzhao Zhang, Shuangge Ma',
    author_email='xmurr@stu.xmu.edu.cn',
    license='MIT',
    packages=find_packages(),
    include_package_data = True,
    package_data={'isfun_data': ['*.pkl'],},
    install_requires=[
        'numpy',
        'pandas',
        'scikit_learn',
        'scipy',
        'seaborn',
        'setuptools',
        'matplotlib'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
    ],
)
