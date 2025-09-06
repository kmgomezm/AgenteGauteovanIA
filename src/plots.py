import matplotlib.pyplot as plt
import pandas as pd

def plot_counts(df: pd.DataFrame, by: str):
    counts = df[by].value_counts().sort_index()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax)
    ax.set_ylabel('Conteos')
    ax.set_title(f'Conteos por {by}')
    return fig
