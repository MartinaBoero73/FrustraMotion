import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from scipy.interpolate import UnivariateSpline
from scipy.stats import zscore
import networkx as nx
from leidenalg import find_partition, RBConfigurationVertexPartition
import igraph as ig
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
import argparse
import sys
import os

warnings.filterwarnings('ignore')


class DynamicClustersAnalyzer:
    """
    Implementación en Python de detect_dynamic_clusters para análisis de frustración dinámica.

    Esta clase replica la funcionalidad de la función R detect_dynamic_clusters(),
    permitiendo detectar módulos de residuos con dinámicas de frustración similares.
    """

    def __init__(self):
        self.results = {}
        self.network = None
        self.clusters = None

    def detect_dynamic_clusters(self,
                                frustration_data: pd.DataFrame,
                                chain_col: str = 'chain',
                                residue_col: str = 'residue',
                                loess_span: float = 0.05,
                                min_frst_range: float = 0.7,
                                filt_mean: float = 0.15,
                                n_components: int = 10,
                                min_corr: float = 0.95,
                                leiden_resolution: float = 1.0,
                                corr_type: str = 'spearman') -> Dict:
        """
        Detecta clusters dinámicos de residuos basado en similitud de frustración.

        Parameters:
        -----------
        frustration_data : pd.DataFrame
            DataFrame avec colonnes 'chain', 'residue', 'frame0', 'frame10', etc.
            Chaque ligne est un résidu, chaque colonne frame contient les valeurs de frustration
        chain_col : str, default 'chain'
            Nom de la colonne contenant les chaînes
        residue_col : str, default 'residue'
            Nom de la colonne contenant les numéros de résidus
        loess_span : float, default 0.05
            Paramètre de lissage pour l'ajustement LOESS
        min_frst_range : float, default 0.7
            Percentile minimum du range de frustration dynamique pour filtrer
        filt_mean : float, default 0.15
            Seuil de filtrage par moyenne de frustration
        n_components : int, default 10
            Nombre de composantes principales à utiliser dans PCA
        min_corr : float, default 0.95
            Seuil minimum de corrélation pour connecter les résidus
        leiden_resolution : float, default 1.0
            Paramètre de résolution pour le clustering de Leiden
        corr_type : str, default 'spearman'
            Type de corrélation ('spearman' ou 'pearson')

        Returns:
        --------
        Dict: Dictionnaire avec les résultats de l'analyse
        """

        print("-----------------------------Loading Data-----------------------------")

        # Identifier les colonnes de frames
        frame_cols = [col for col in frustration_data.columns
                      if col.startswith('frame') and col not in [chain_col, residue_col]]
        frame_cols = sorted(frame_cols, key=lambda x: int(x.replace('frame', '')))

        print(f"Detected frame columns: {len(frame_cols)} frames")

        # Créer les noms de résidus
        residue_names = (frustration_data[chain_col].astype(str) + '_' +
                         frustration_data[residue_col].astype(str))

        # Extraire seulement les données de frustration (colonnes frame)
        frustration_matrix = frustration_data[frame_cols]
        frustration_matrix.index = residue_names

        print(f"Frustration matrix: {frustration_matrix.shape[0]} residues × {frustration_matrix.shape[1]} frames")

        print("-----------------------------Model fitting and filtering-----------------------------")

        # Ajuster modèles LOESS et calculer statistiques
        fitted_data = []
        frst_ranges = []
        means = []
        sds = []

        for i, (res_name, row) in enumerate(frustration_matrix.iterrows()):
            # Créer modèle LOESS en utilisant spline comme approximation
            frames = np.arange(len(row))
            frustration_values = row.values

            # Vérifier s'il y a des valeurs manquantes
            if np.any(pd.isna(frustration_values)):
                print(f"Warning: Missing values detected for {res_name}")
                frustration_values = pd.Series(frustration_values).interpolate().fillna(method='bfill').fillna(
                    method='ffill').values

            # Utiliser spline comme approximation à LOESS
            try:
                # Convertir loess_span en facteur de lissage pour spline
                smoothing_factor = len(frames) * loess_span * 100  # Ajustement pour spline
                spline = UnivariateSpline(frames, frustration_values, s=smoothing_factor)
                fitted_values = spline(frames)
            except:
                # Fallback: utiliser moyenne mobile
                window_size = max(1, int(len(frames) * loess_span * 10))
                fitted_values = pd.Series(frustration_values).rolling(
                    window=window_size, center=True, min_periods=1
                ).mean().values

            fitted_data.append(fitted_values)
            frst_ranges.append(np.max(fitted_values) - np.min(fitted_values))
            means.append(np.mean(fitted_values))
            sds.append(np.std(fitted_values))

        fitted_data = np.array(fitted_data)
        frst_ranges = np.array(frst_ranges)
        means = np.array(means)
        sds = np.array(sds)

        # Filtrer résidus basé sur range dynamique et moyenne
        range_threshold = np.percentile(frst_ranges, min_frst_range * 100)
        mean_filter = (means < -filt_mean) | (means > filt_mean)
        range_filter = frst_ranges > range_threshold

        selected_residues = range_filter & mean_filter

        print(f"Selected residues: {np.sum(selected_residues)} out of {len(residue_names)}")

        if np.sum(selected_residues) < 2:
            print("Warning: Too few residues pass the filters. Adjust parameters.")
            return {}

        # Filtrer données
        filtered_data = frustration_matrix.iloc[selected_residues]
        filtered_names = [residue_names[i] for i in range(len(residue_names)) if selected_residues[i]]

        print("-----------------------------Principal Component Analysis-----------------------------")

        # PCA - aplicar directamente a los residuos (filas) usando sus series temporales
        scaler = StandardScaler()
        # filtered_data shape: (n_residuos, n_frames)
        scaled_data = scaler.fit_transform(filtered_data)  # Escalar cada residuo

        # Aplicar PCA para reducir dimensionalidad temporal
        pca = PCA(n_components=min(n_components, scaled_data.shape[1], scaled_data.shape[0]))
        pca_coords = pca.fit_transform(scaled_data)  # Shape: (n_residuos, n_components)

        print(f"Explained variance by {pca.n_components_} components: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"PCA coordinates shape: {pca_coords.shape}")  # Debug

        print("-----------------------------Correlation calculation-----------------------------")

        # Calcular matriz de correlación usando las coordenadas PCA
        corr_matrix = np.zeros((len(filtered_names), len(filtered_names)))
        p_values = np.zeros((len(filtered_names), len(filtered_names)))

        for i in range(len(filtered_names)):
            for j in range(i + 1, len(filtered_names)):
                if corr_type.lower() == 'spearman':
                    corr, p_val = spearmanr(pca_coords[i], pca_coords[j])
                else:
                    corr, p_val = pearsonr(pca_coords[i], pca_coords[j])

                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                p_values[i, j] = p_val
                p_values[j, i] = p_val

        # Filtrer correlaciones
        adjacency_matrix = np.zeros_like(corr_matrix)
        significant_corr = (np.abs(corr_matrix) >= min_corr) & (p_values <= 0.05)
        adjacency_matrix[significant_corr] = corr_matrix[significant_corr]

        print("-----------------------------Graph construction-----------------------------")

        # Crear grafo con igraph
        # Convertir a formato de aristas
        edges = []
        weights = []

        for i in range(len(filtered_names)):
            for j in range(i + 1, len(filtered_names)):
                if adjacency_matrix[i, j] != 0:
                    edges.append((i, j))
                    weights.append(abs(adjacency_matrix[i, j]))

        if not edges:
            print("Warning: No significant connections found.")
            return {}

        # Crear grafo igraph
        g = ig.Graph()
        g.add_vertices(len(filtered_names))
        g.add_edges(edges)
        g.es['weight'] = weights
        g.vs['name'] = filtered_names

        # Remover vértices aislados
        isolated_vertices = [v.index for v in g.vs if v.degree() == 0]
        g.delete_vertices(isolated_vertices)

        # Actualizar nombres después de eliminar vértices aislados
        remaining_names = [name for i, name in enumerate(filtered_names) if i not in isolated_vertices]

        print("-----------------------------Leiden clustering-----------------------------")

        if len(g.vs) < 2:
            print("Warning: Too few connected vertices for clustering.")
            return {}

        # Aplicar clustering de Leiden
        partition = find_partition(g, RBConfigurationVertexPartition,
                                   resolution_parameter=leiden_resolution)

        cluster_labels = list(partition.membership)

        print(f"Number of clusters found: {len(set(cluster_labels))}")

        # Preparar resultados
        cluster_df = pd.DataFrame({
            'Residue': remaining_names,
            'Cluster': cluster_labels
        })

        # Statistiques par résidu (des données originales)
        original_residue_names = list(residue_names)
        stats_df = pd.DataFrame({
            'Residue': original_residue_names,
            'Chain': frustration_data[chain_col].values,
            'ResidueNum': frustration_data[residue_col].values,
            'Mean': means,
            'Sd': sds,
            'FrstRange': frst_ranges,
            'Selected': selected_residues
        })

        # Combinar con información de clusters
        cluster_df = cluster_df.merge(stats_df, on='Residue', how='left')

        self.results = {
            'clusters': cluster_df,
            'graph': g,
            'partition': partition,
            'correlation_matrix': corr_matrix,
            'adjacency_matrix': adjacency_matrix,
            'pca_coords': pca_coords,
            'pca_model': pca,
            'fitted_data': fitted_data,
            'parameters': {
                'loess_span': loess_span,
                'min_frst_range': min_frst_range,
                'filt_mean': filt_mean,
                'n_components': n_components,
                'min_corr': min_corr,
                'leiden_resolution': leiden_resolution,
                'corr_type': corr_type
            },
            'statistics': stats_df,
            'filtered_residues': filtered_names
        }

        self.network = g
        self.clusters = cluster_df

        print("Process completed successfully!")
        return self.results

    def plot_network(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 10)):
        """Plot the network of clustered residues"""
        if self.results is None:
            print("Run detect_dynamic_clusters() first")
            return None

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if self.network and len(self.network.vs) > 0:
            try:
                pos = self.network.layout('fr')
                colors = plt.cm.Set3(np.array(self.results['partition'].membership))

                # Create network plot manually as ig.plot may have issues
                edge_list = [(e.source, e.target) for e in self.network.es]
                pos_dict = {i: pos[i] for i in range(len(pos))}

                # Draw edges
                for edge in edge_list:
                    x_coords = [pos_dict[edge[0]][0], pos_dict[edge[1]][0]]
                    y_coords = [pos_dict[edge[0]][1], pos_dict[edge[1]][1]]
                    ax.plot(x_coords, y_coords, 'k-', alpha=0.3, linewidth=0.8)

                # Draw nodes
                x_pos = [pos_dict[i][0] for i in range(len(pos_dict))]
                y_pos = [pos_dict[i][1] for i in range(len(pos_dict))]
                scatter = ax.scatter(x_pos, y_pos, c=colors, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)

                # Add residue names as labels
                residue_names = self.network.vs['name']
                for i, (x, y) in enumerate(zip(x_pos, y_pos)):
                    ax.annotate(residue_names[i], (x, y),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=9, alpha=0.8,
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='white', alpha=0.8, edgecolor='gray'))

                ax.set_title('Network of Clustered Residues', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel('X Coordinate', fontsize=12)
                ax.set_ylabel('Y Coordinate', fontsize=12)
                ax.grid(True, alpha=0.2)

                # Add cluster legend
                unique_clusters = sorted(set(self.results['partition'].membership))
                legend_elements = [plt.scatter([], [], c=plt.cm.Set3(i), s=60, alpha=0.8,
                                               edgecolors='black', linewidth=0.5,
                                               label=f'Cluster {i}')
                                   for i in unique_clusters]
                ax.legend(handles=legend_elements, title='Clusters', loc='upper right',
                          framealpha=0.9, fontsize=10)

            except Exception as e:
                ax.text(0.5, 0.5, f'Error in network visualization:\n{str(e)}',
                        transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title('Network of Residues (Error)', fontsize=16)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network plot saved to: {save_path}")

        plt.show()
        return fig

    def plot_frustration_stats(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 10)):
        """Plot frustration statistics with enhanced visualization"""
        if self.results is None or 'statistics' not in self.results:
            print("Run detect_dynamic_clusters() first")
            return None

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        stats = self.results['statistics']
        selected = stats['Selected']

        # Get filtering thresholds
        params = self.results['parameters']
        range_threshold = np.percentile(stats['FrstRange'], params['min_frst_range'] * 100)
        mean_threshold = params['filt_mean']

        # Plot non-selected points (gray)
        gray_points = ax.scatter(stats['Mean'][~selected], stats['FrstRange'][~selected],
                                 c='lightgray', alpha=0.6, s=50, edgecolors='darkgray',
                                 linewidth=0.5, label='Non-selected residues')

        # Plot selected points (red)
        red_points = ax.scatter(stats['Mean'][selected], stats['FrstRange'][selected],
                                c='red', alpha=0.8, s=60, edgecolors='darkred',
                                linewidth=0.8, label='Selected residues')

        # Add dotted lines for thresholds
        ax.axhline(y=range_threshold, color='red', linestyle=':', linewidth=2,
                   alpha=0.8, label=f'Range threshold ({params["min_frst_range"]:.0%} percentile)')
        ax.axvline(x=mean_threshold, color='red', linestyle=':', linewidth=2, alpha=0.8)
        ax.axvline(x=-mean_threshold, color='red', linestyle=':', linewidth=2, alpha=0.8,
                   label=f'Mean threshold (±{mean_threshold})')

        # Add residue names for selected points (red points)
        selected_stats = stats[selected]
        for idx, row in selected_stats.iterrows():
            ax.annotate(row['Residue'],
                        (row['Mean'], row['FrstRange']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.9, color='darkred',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.8,
                                  edgecolor='red', linewidth=0.5))

        ax.set_xlabel('Frustration Mean', fontsize=14)
        ax.set_ylabel('Frustration Range', fontsize=14)
        ax.set_title('Frustration Statistics', fontsize=16, fontweight='bold', pad=20)
        ax.legend(framealpha=0.9, fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        n_total = len(stats)
        n_selected = np.sum(selected)
        stats_text = f'Total residues: {n_total}\nSelected residues: {n_selected} ({n_selected / n_total:.1%})'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Frustration statistics plot saved to: {save_path}")

        plt.show()
        return fig

    def plot_cluster_distribution(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)):
        """Plot cluster size distribution"""
        if self.clusters is None:
            print("Run detect_dynamic_clusters() first")
            return None

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        cluster_counts = self.clusters['Cluster'].value_counts().sort_index()
        bars = ax.bar(range(len(cluster_counts)), cluster_counts.values,
                      color=plt.cm.Set3(np.arange(len(cluster_counts))),
                      alpha=0.8, edgecolor='black', linewidth=0.8)

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Number of Residues', fontsize=12)
        ax.set_title('Cluster Distribution', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(range(len(cluster_counts)))
        ax.set_xticklabels(cluster_counts.index)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster distribution plot saved to: {save_path}")

        plt.show()
        return fig

    def plot_pca_variance(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (10, 6)):
        """Plot PCA explained variance"""
        if 'pca_model' not in self.results:
            print("Run detect_dynamic_clusters() first")
            return None

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        pca_var = self.results['pca_model'].explained_variance_ratio_
        cumulative_var = np.cumsum(pca_var)

        bars = ax.bar(range(1, len(pca_var) + 1), pca_var,
                      color='steelblue', alpha=0.7, edgecolor='navy', linewidth=0.8,
                      label='Individual')

        # Add cumulative line
        ax2 = ax.twinx()
        line = ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var,
                        'ro-', color='red', alpha=0.8, linewidth=2, markersize=6,
                        label='Cumulative')

        # Add percentage labels
        for i, (bar, var) in enumerate(zip(bars, pca_var)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{var:.1%}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=12, color='red')
        ax.set_title('PCA Explained Variance', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA variance plot saved to: {save_path}")

        plt.show()
        return fig

    def plot_all(self, save_dir: Optional[str] = None):
        """Plot all visualizations separately"""
        print("Generating all plots...")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Network plot
        network_path = os.path.join(save_dir, "network_plot.png") if save_dir else None
        self.plot_network(save_path=network_path)

        # Frustration statistics
        stats_path = os.path.join(save_dir, "frustration_stats.png") if save_dir else None
        self.plot_frustration_stats(save_path=stats_path)

        # Cluster distribution
        cluster_path = os.path.join(save_dir, "cluster_distribution.png") if save_dir else None
        self.plot_cluster_distribution(save_path=cluster_path)

        # PCA variance
        pca_path = os.path.join(save_dir, "pca_variance.png") if save_dir else None
        self.plot_pca_variance(save_path=pca_path)

    def plot_clusters(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 10)):
        """Legacy method - now redirects to plot_all()"""
        print("Note: plot_clusters() now shows individual plots. Use plot_all() for saving all plots.")
        self.plot_all()

    def get_cluster_info(self, cluster_ids: Union[List[int], str] = 'all') -> pd.DataFrame:
        """
        Obtener información de clusters específicos

        Parameters:
        -----------
        cluster_ids : List[int] or 'all'
            IDs de clusters a obtener información

        Returns:
        --------
        pd.DataFrame: Información de residuos en los clusters especificados
        """
        if self.clusters is None:
            print("Run detect_dynamic_clusters() first")
            return pd.DataFrame()

        if cluster_ids == 'all':
            return self.clusters
        else:
            return self.clusters[self.clusters['Cluster'].isin(cluster_ids)]

    def save_results(self, output_dir: str = "cluster_results"):
        """
        Guardar todos los resultados en archivos

        Parameters:
        -----------
        output_dir : str
            Directorio donde guardar los resultados
        """
        if self.results is None:
            print("No results to save. Run detect_dynamic_clusters() first.")
            return

        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Guardar clusters
        clusters_file = os.path.join(output_dir, "clusters.csv")
        if self.clusters is not None:
            self.clusters.to_csv(clusters_file, index=False)
            print(f"Cluster information saved to: {clusters_file}")

        # Guardar estadísticas
        stats_file = os.path.join(output_dir, "statistics.csv")
        if 'statistics' in self.results:
            self.results['statistics'].to_csv(stats_file, index=False)
            print(f"Statistics saved to: {stats_file}")

        # Guardar parámetros
        params_file = os.path.join(output_dir, "parameters.txt")
        if 'parameters' in self.results:
            with open(params_file, 'w') as f:
                f.write("Parameters used:\n")
                f.write("================\n\n")
                for key, value in self.results['parameters'].items():
                    f.write(f"{key}: {value}\n")
            print(f"Parameters saved to: {params_file}")

        # Guardar gráficos
        plots_dir = os.path.join(output_dir, "plots")
        self.plot_all(save_dir=plots_dir)

        # Resumen
        summary_file = os.path.join(output_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write("DYNAMIC CLUSTERS ANALYSIS SUMMARY\n")
            f.write("=================================\n\n")

            if 'statistics' in self.results:
                total_residues = len(self.results['statistics'])
                selected_residues = np.sum(self.results['statistics']['Selected'])
                f.write(f"Total residues: {total_residues}\n")
                f.write(f"Selected residues: {selected_residues}\n")

            if self.clusters is not None:
                n_clusters = self.clusters['Cluster'].nunique()
                f.write(f"Number of clusters: {n_clusters}\n\n")

                # Resumen por cluster
                f.write("Summary per cluster:\n")
                f.write("-------------------\n")
                cluster_summary = self.clusters.groupby('Cluster').agg({
                    'Residue': 'count',
                    'Chain': lambda x: ', '.join(x.astype(str).unique()),
                    'Mean': ['mean', 'std'],
                    'FrstRange': ['mean', 'std']
                }).round(3)
                f.write(cluster_summary.to_string())

        print(f"Summary saved to: {summary_file}")
        print(f"\nAll results saved in directory: {output_dir}")


def main():
    """Main function to run from command line"""

    parser = argparse.ArgumentParser(
        description="Dynamic frustration clusters analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python cluster.py data.csv
  python cluster.py data.xlsx --chain_col chain --residue_col residue
  python cluster.py data.csv --min_corr 0.8 --output results_folder
  python cluster.py data.csv --help-params
        """
    )

    # Required argument
    parser.add_argument('file', help='Path to data file (CSV or Excel)')

    # Column arguments
    parser.add_argument('--chain_col', default='chain',
                        help='Name of chain column (default: chain)')
    parser.add_argument('--residue_col', default='residue',
                        help='Name of residue column (default: residue)')

    # Algorithm parameters
    parser.add_argument('--loess_span', type=float, default=0.05,
                        help='LOESS smoothing parameter (default: 0.05)')
    parser.add_argument('--min_frst_range', type=float, default=0.7,
                        help='Minimum percentile of frustration range (default: 0.7)')
    parser.add_argument('--filt_mean', type=float, default=0.15,
                        help='Mean filtering threshold (default: 0.15)')
    parser.add_argument('--n_components', type=int, default=10,
                        help='Number of PCA components (default: 10)')
    parser.add_argument('--min_corr', type=float, default=0.95,
                        help='Minimum correlation for connections (default: 0.95)')
    parser.add_argument('--leiden_resolution', type=float, default=1.0,
                        help='Leiden clustering resolution (default: 1.0)')
    parser.add_argument('--corr_type', choices=['spearman', 'pearson'], default='spearman',
                        help='Correlation type (default: spearman)')

    # Output arguments
    parser.add_argument('--output', '-o', default='cluster_results',
                        help='Output directory (default: cluster_results)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Do not show interactive plots')

    # Additional help
    parser.add_argument('--help-params', action='store_true',
                        help='Show detailed parameter explanation')

    args = parser.parse_args()

    if args.help_params:
        print("""
DETAILED PARAMETER EXPLANATION:
==============================

--loess_span (0.01-0.2): 
    Smoothing control. Lower values = less smoothing.

--min_frst_range (0.5-0.9):
    Percentile for filtering by dynamic range. 0.7 = only residues with range 
    in the top 30% most variable.

--filt_mean (0.05-0.3):
    Absolute threshold for filtering by frustration mean. Only residues with
    |mean| > threshold.

--n_components (5-20):
    PCA dimensions. More components = more information but more noise.

--min_corr (0.7-0.99):
    Minimum correlation between residues to form connections. High values
    = stricter clusters.

--leiden_resolution (0.1-2.0):
    Cluster size control. High values = more small clusters.

TIPS:
- For low-noise data: high min_corr (>0.9)
- To find more clusters: high leiden_resolution (>1.5)
- For very variable data: low filt_mean (<0.1)
        """)
        return

    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' does not exist.")
        sys.exit(1)

    print(f"Analyzing file: {args.file}")
    print("=" * 50)

    try:
        # Load data
        if args.file.endswith('.csv'):
            df = pd.read_csv(args.file)
        elif args.file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(args.file)
        else:
            print("Error: Unsupported file format. Use CSV or Excel.")
            sys.exit(1)

        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Detected columns: {list(df.columns)}")

        # Check required columns
        if args.chain_col not in df.columns:
            print(f"Error: Column '{args.chain_col}' not found in data.")
            sys.exit(1)

        if args.residue_col not in df.columns:
            print(f"Error: Column '{args.residue_col}' not found in data.")
            sys.exit(1)

        # Check frame columns
        frame_cols = [col for col in df.columns if col.startswith('frame')]
        if len(frame_cols) == 0:
            print("Error: No columns starting with 'frame' found.")
            sys.exit(1)

        print(f"Detected frame columns: {len(frame_cols)}")

        # Create analyzer
        analyzer = DynamicClustersAnalyzer()

        # Run analysis
        results = analyzer.detect_dynamic_clusters(
            frustration_data=df,
            chain_col=args.chain_col,
            residue_col=args.residue_col,
            loess_span=args.loess_span,
            min_frst_range=args.min_frst_range,
            filt_mean=args.filt_mean,
            n_components=args.n_components,
            min_corr=args.min_corr,
            leiden_resolution=args.leiden_resolution,
            corr_type=args.corr_type
        )

        if not results:
            print("Error: Could not generate results. Check parameters.")
            sys.exit(1)

        # Show summary
        print("\n" + "=" * 50)
        print("RESULTS SUMMARY")
        print("=" * 50)

        if analyzer.clusters is not None:
            n_clusters = analyzer.clusters['Cluster'].nunique()
            print(f"Clusters found: {n_clusters}")

            cluster_summary = analyzer.clusters.groupby('Cluster').agg({
                'Residue': 'count',
                'Chain': lambda x: ', '.join(x.astype(str).unique()),
                'Mean': 'mean',
                'FrstRange': 'mean'
            }).round(3)
            print("\nSummary per cluster:")
            print(cluster_summary)

        # Save results
        analyzer.save_results(args.output)

        # Show plots if requested
        if not args.no_plot:
            print("\nShowing plots...")
            analyzer.plot_all()

        print(f"\n✓ Analysis completed successfully!")
        print(f"✓ Results saved to: {args.output}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()