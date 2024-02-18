import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def ispls_plot(x, type):
    betahat = x.betahat
    L = len(betahat)
    p = betahat[list(betahat.keys())[0]].shape[0]
    q = betahat[list(betahat.keys())[0]].shape[1]
    loading_trace = x.loading_trace
    iter = loading_trace.shape[1]
    if type == "path":
        for l in range(L):
            plt.figure()
            plt.plot(np.transpose(loading_trace[(l * p):((l + 1) * p), :]))
            plt.title(f"Dataset {l+1}\nConvergence path of elements in vector w")
            plt.xlabel("Number of iterations")
            plt.ylabel("Weight")
            plt.show()
    if type == "loading":
        for l in range(L):
            plt.figure()
            plt.scatter(x = range(1, p+1), y = loading_trace[(l * p):((l + 1) * p), iter-1])
            plt.title(f"Dataset {l+1}\nThe first direction vector")
            plt.xlabel("Dimension")
            plt.ylabel("Value")
            plt.show()
    if type == "heatmap" and q != 1:
        for l in range(L):
            plt.figure()
            sns.heatmap(np.transpose(betahat[list(betahat.keys())[l]]), cmap='Reds')
            plt.title(f"Dataset {l+1}: p\nHeatmap of coefficient β[PLS]")
            plt.show()
    if type == "heatmap" and q == 1:
        betahat_matrix = np.zeros((p, L))
        for l in range(L):
            betahat_matrix[:, l] = betahat[list(betahat.keys())[l]]
        plt.figure()
        sns.heatmap(np.transpose(betahat_matrix), cmap='Reds')
        plt.title("p\nHeatmap of coefficient β[PLS]")
        plt.show()
