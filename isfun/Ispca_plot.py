import matplotlib.pyplot as plt
import numpy as np

def ispca_plot(x, type):
    loadings = x.eigenvector
    L = loadings.shape[1]
    p = loadings.shape[0]
    loading_trace = x.loading_trace
    iter = loading_trace.shape[1]
    if type == "path":
        for l in range(L):
            plt.figure()
            plt.plot(np.transpose(loading_trace[(l * p):((l + 1) * p), :]))
            plt.title(f"Dataset {l+1}\nConvergence path of elements in vector u")
            plt.xlabel("Number of iterations")
            plt.ylabel("Weight")
            plt.show()
    if type == "loading":
        for l in range(L):
            plt.figure()
            plt.scatter(range(1, p+1), loading_trace[(l * p):((l + 1) * p), iter-1])
            plt.title(f"The first canonical vector u of dataset {l+1}")
            plt.xlabel("Dimension")
            plt.ylabel("Value")
            plt.show()