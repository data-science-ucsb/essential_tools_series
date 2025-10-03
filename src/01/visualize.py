import numpy as np
import matplotlib.pyplot as plt

def show_fit(x, y, m, b, title="Linear Fit", xlabel="x", ylabel="y"):
    plt.figure()
    plt.scatter(x, y, label="data")
    # Draw a smooth line across the x-range
    x_line = np.linspace(np.min(x), np.max(x), 200)
    plt.plot(x_line, m * x_line + b, label="best-fit line", color="red")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()