import matplotlib.pyplot as plt
import pandas as pd


def plot_histogram(data, title, xlabel, ylabel, bins=30, color="steelblue"):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, color=color, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_barh(df: pd.DataFrame, x_col: str, y_col: str, title: str, color="coral"):
    plt.figure(figsize=(8, 5))
    plt.barh(df[y_col], df[x_col], color=color)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()
    plt.show()
