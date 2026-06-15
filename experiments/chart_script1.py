import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

METHOD_ORDER = ['Naive', 'Bootstrap', 'RSBS']

def generate_publication_ready_plots(data_file, n_nodes):
    if not os.path.exists(data_file):
        print(f"Error: '{data_file}' not found.")
        return
        
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    # Dynamic calculations
    df['pct_perturbed'] = (df['n_perturbed_nodes'] / n_nodes * 100).round(1).astype(str) + '%'
    df['positives_label'] = df['n_positives'].astype(str)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    COLOR_MAP = {'Naive': '#5F4690', 'Bootstrap': '#1D6996', 'RSBS': '#E69F00'}
    available_methods = [m for m in METHOD_ORDER if m in df['method'].unique()]
    current_palette = {m: COLOR_MAP[m] for m in available_methods}

    # --- PLOT 1: Learning Curves with Dynamic Labeling ---
    print("Generating 'fig1_learning_curves.png'...")
    
    # Check for symmetry
    is_symmetric = (df['n1'] == df['n2']).all()
    
    if is_symmetric:
        x_label = "Sample Size"
        setup_title = "Experimental Setup: Symmetric (N1 = N2)"
    else:
        x_label = "Primary Sample Size (N1)"
        setup_title = "Experimental Setup: Heterogeneous Sample Ratios"

    g1 = sns.relplot(
        data=df, x='n1', y='aupr', hue='method', col='edge_prob',
        kind='line', palette=current_palette, hue_order=available_methods,
        markers=True, dashes=False, errorbar=('ci', 95), 
        facet_kws={'sharex': False, 'sharey': True},
        height=4.5, aspect=1.2, linewidth=2.5, markersize=9
    )
    
    g1.set_axis_labels(x_label, "Mean AUPR")
    g1.set_titles(col_template="Density: {col_name}", fontweight='bold')
    g1.fig.subplots_adjust(top=0.85)
    g1.fig.suptitle(setup_title, fontsize=16, fontweight='bold')
    
    sns.move_legend(g1, "upper center", bbox_to_anchor=(0.5, -0.15), 
                    ncol=len(available_methods), frameon=False, title=None)
    g1.savefig("fig1_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 2: Topological Stress Test ---
    print("Generating 'fig2_topological_stress.png'...")
    sparsity_levels = sorted(df['edge_prob'].unique())
    fig2, axes2 = plt.subplots(1, len(sparsity_levels), figsize=(6 * len(sparsity_levels), 5), sharey=True)
    if len(sparsity_levels) == 1: axes2 = [axes2]
    
    for i, sparsity in enumerate(sparsity_levels):
        sns.barplot( 
            data=df[df['edge_prob'] == sparsity].sort_values('n_perturbed_nodes'),
            x='pct_perturbed', y='aupr', hue='method', hue_order=available_methods,
            palette=current_palette, ax=axes2[i], capsize=.1, errorbar=('ci', 95)
        )
        axes2[i].set_title(f"Density: {sparsity}", fontsize=14, fontweight='bold')
        axes2[i].set_xlabel("Perturbed (%)")
        axes2[i].set_ylabel("Mean AUPR" if i == 0 else "")
        if axes2[i].get_legend() is not None: axes2[i].get_legend().remove()
            
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
                ncol=len(available_methods), frameon=False, fontsize=13)
    plt.tight_layout()
    fig2.savefig("fig2_topological_stress.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 3: Intervention Complexity ---
    print("Generating 'fig3_intervention_complexity.png'...")
    df_pos = df[df['edge_prob'] == df['edge_prob'].max()] 
    
    if not df_pos.empty:
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(data=df_pos.sort_values('n_positives'), x='positives_label', y='precision', 
                    hue='method', hue_order=available_methods,
                    palette=current_palette, ax=axes3[0], capsize=.1, errorbar=('ci', 95))
        axes3[0].set_title("Precision vs Complexity", fontsize=14, fontweight='bold')
        axes3[0].set_xlabel("Number of Intervention Targets")
        
        sns.barplot(data=df_pos.sort_values('n_positives'), x='positives_label', y='recall', 
                    hue='method', hue_order=available_methods,
                    palette=current_palette, ax=axes3[1], capsize=.1, errorbar=('ci', 95))
        axes3[1].set_title("Recall vs Complexity", fontsize=14, fontweight='bold')
        axes3[1].set_xlabel("Number of Intervention Targets")
        
        for ax in axes3:
            if ax.get_legend(): ax.get_legend().remove()
        
        handles, labels = axes3[0].get_legend_handles_labels()
        fig3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), 
                    ncol=len(available_methods), frameon=False, fontsize=13)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        fig3.savefig("fig3_intervention_complexity.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("Process complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication plots.")
    parser.add_argument("--file", required=True, help="Path to the master CSV file.")
    parser.add_argument("--nodes", type=int, default=15, help="Number of nodes.")
    args = parser.parse_args()
    generate_publication_ready_plots(args.file, args.nodes)