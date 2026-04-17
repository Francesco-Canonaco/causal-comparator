import pandas as pd
import plotly.express as px
import plotly.io as pio

# Set default template 
pio.templates.default = "plotly_white"

# Define the consistent color map globally for the module
PRISM_COLORS = px.colors.qualitative.Prism
COLOR_MAP = {
    'Naive': PRISM_COLORS[0],
    'Bootstrap': PRISM_COLORS[1],
    'RSBS': PRISM_COLORS[2]
}
METHODS_ORDER = ['Naive', 'Bootstrap', 'RSBS']

def plot_ranking_metrics(df, save_path=None):
    """
    Generates boxplots for AUC-ROC and AUPR.
    """
    ranking_df = df.melt(
        id_vars=['method', 'seed'], 
        value_vars=['auc_roc', 'aupr'], 
        var_name='Metric', value_name='Value'
    )

    fig = px.box(
        ranking_df, x='Metric', y='Value', color='method', 
        category_orders={"method": METHODS_ORDER},
        points="all", 
        color_discrete_map=COLOR_MAP,
        title='Causal Discovery Performance: Ranking Metrics',
        labels={'Value': 'Score', 'method': 'Method'}
    )

    fig.update_layout(yaxis_range=[0, 1.05], boxmode='group')
    
    if save_path:
        fig.write_html(save_path)
    return fig

def plot_classification_metrics(df, save_path=None):
    """
    Generates boxplots for F1, Precision, and Recall.
    """
    class_df = df.melt(
        id_vars=['method', 'seed'], 
        value_vars=['best_f1', 'precision', 'recall'], 
        var_name='Metric', value_name='Value'
    )

    fig = px.box(
        class_df, x='Metric', y='Value', color='method', 
        category_orders={"method": METHODS_ORDER},
        color_discrete_map=COLOR_MAP,
        title='Classification Performance (Optimized Thresholds)',
        labels={'Value': 'Score', 'method': 'Method'}
    )

    fig.update_layout(yaxis_range=[0, 1.05], boxmode='group')

    if save_path:
        fig.write_html(save_path)
    return fig

def plot_summary_performance(df, save_path=None):
    """
    Generates a grouped bar chart of mean scores.
    """
    summary = df.groupby('method')[['auc_roc', 'aupr', 'best_f1', 'precision', 'recall']].mean()
    summary = summary.reindex(METHODS_ORDER).reset_index()
    
    summary_melted = summary.melt(
        id_vars='method', 
        var_name='Metric', value_name='Mean Value'
    )

    fig = px.bar(
        summary_melted, x='Metric', y='Mean Value', color='method', 
        barmode='group',
        color_discrete_map=COLOR_MAP,
        title='Average Performance Comparison (All Seeds)',
        text_auto='.3f'
    )

    fig.update_layout(yaxis_range=[0, 1.1])

    if save_path:
        fig.write_html(save_path)
    return fig