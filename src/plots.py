import matplotlib.pyplot as plt
import pandas as pd

def plot_counts(df: pd.DataFrame, by: str):
    if df is None or df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No hay datos para mostrar', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Conteos por {by}')
        return fig

    counts = df[by].value_counts().sort_index()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax)
    ax.set_ylabel('Conteos')
    ax.set_title(f'Conteos por {by}')
    return fig
