# plot_utils.py
# ------------------------------
# PURPOSE: Utility functions for plotting data

import matplotlib.pyplot as plt

def plot_histogram(data, column, bins=50, color="blue", title="Histogram"):
    plt.figure(figsize=(8, 6))
    plt.hist(data[column], bins=bins, color=color, edgecolor="black")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()
