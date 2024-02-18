import matplotlib.pyplot as plt
import numpy as np

def iscca_plot(x, type):
    loading_x = x.loading_x
    loading_y = x.loading_y
    L = loading_x.shape[1]
    p = loading_x.shape[0]
    q = loading_y.shape[0]
    loading_trace_u = x.loading_trace_u
    loading_trace_v = x.loading_trace_v
    iter_u = loading_trace_u.shape[1]
    iter_v = loading_trace_v.shape[1]
    if type == "path":
        for l in range(L):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(np.transpose(loading_trace_u[(l * p):((l + 1) * p), :]))
            plt.title(f"Dataset {l+1}\nConvergence path of elements in vector u")
            plt.xlabel("Number of iterations")
            plt.ylabel("Weight")
            plt.subplot(1, 2, 2)
            plt.plot(np.transpose(loading_trace_v[(l * q):((l + 1) * q), :]))
            plt.title(f"Dataset {l+1}\nConvergence path of elements in vector v")
            plt.xlabel("Number of iterations")
            plt.ylabel("Weight")
            plt.tight_layout()
            plt.show()
    if type == "loading":
        for l in range(L):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(range(1, p+1), loading_trace_u[(l * p):((l + 1) * p), iter_u-1])
            plt.title(f"The first canonical vector u of dataset {l+1}")
            plt.xlabel("Dimension")
            plt.ylabel("Value")
            plt.subplot(1, 2, 2)
            plt.scatter(range(1, p+1), loading_trace_v[(l * p):((l + 1) * p), iter_v-1])
            plt.title(f"The first canonical vector v of dataset {l+1}")
            plt.xlabel("Dimension")
            plt.ylabel("Value")
            plt.tight_layout()
            plt.show()