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
        
        print("-----------------------------Chargement des données-----------------------------")
        
        # Identifier les colonnes de frames
        frame_cols = [col for col in frustration_data.columns 
                     if col.startswith('frame') and col not in [chain_col, residue_col]]
        frame_cols = sorted(frame_cols, key=lambda x: int(x.replace('frame', '')))
        
        print(f"Colonnes de frames détectées: {len(frame_cols)} frames")
        
        # Créer les noms de résidus
        residue_names = (frustration_data[chain_col].astype(str) + '_' + 
                        frustration_data[residue_col].astype(str))
        
        # Extraire seulement les données de frustration (colonnes frame)
        frustration_matrix = frustration_data[frame_cols]
        frustration_matrix.index = residue_names
        
        print(f"Matrice de frustration: {frustration_matrix.shape[0]} résidus × {frustration_matrix.shape[1]} frames")
        
        print("-----------------------------Ajustement de modèle et filtrage-----------------------------")
        
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
                print(f"Attention: Valeurs manquantes détectées pour {res_name}")
                frustration_values = pd.Series(frustration_values).interpolate().fillna(method='bfill').fillna(method='ffill').values
            
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
        
        print(f"Résidus sélectionnés: {np.sum(selected_residues)} sur {len(residue_names)}")
        
        if np.sum(selected_residues) < 2:
            print("Attention: Trop peu de résidus passent les filtres. Ajuster les paramètres.")
            return {}
        
        # Filtrer données
        filtered_data = frustration_matrix.iloc[selected_residues]
        filtered_names = [residue_names[i] for i in range(len(residue_names)) if selected_residues[i]]
        
        print("-----------------------------Análisis de componentes principales-----------------------------")
        
        # PCA - aplicar directamente a los residuos (filas) usando sus series temporales
        scaler = StandardScaler()
        # filtered_data shape: (n_residuos, n_frames)
        scaled_data = scaler.fit_transform(filtered_data)  # Escalar cada residuo
        
        # Aplicar PCA para reducir dimensionalidad temporal
        pca = PCA(n_components=min(n_components, scaled_data.shape[1], scaled_data.shape[0]))
        pca_coords = pca.fit_transform(scaled_data)  # Shape: (n_residuos, n_components)
        
        print(f"Varianza explicada por {pca.n_components_} componentes: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"Coordenadas PCA shape: {pca_coords.shape}")  # Debug
        
        print("-----------------------------Cálculo de correlaciones-----------------------------")
        
        # Calcular matriz de correlación usando las coordenadas PCA
        corr_matrix = np.zeros((len(filtered_names), len(filtered_names)))
        p_values = np.zeros((len(filtered_names), len(filtered_names)))
        
        for i in range(len(filtered_names)):
            for j in range(i+1, len(filtered_names)):
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
        
        print("-----------------------------Construcción de grafo-----------------------------")
        
        # Crear grafo con igraph
        # Convertir a formato de aristas
        edges = []
        weights = []
        
        for i in range(len(filtered_names)):
            for j in range(i+1, len(filtered_names)):
                if adjacency_matrix[i, j] != 0:
                    edges.append((i, j))
                    weights.append(abs(adjacency_matrix[i, j]))
        
        if not edges:
            print("Advertencia: No se encontraron conexiones significativas.")
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
        
        print("-----------------------------Clustering de Leiden-----------------------------")
        
        if len(g.vs) < 2:
            print("Advertencia: Muy pocos vértices conectados para clustering.")
            return {}
        
        # Aplicar clustering de Leiden
        partition = find_partition(g, RBConfigurationVertexPartition, 
                                 resolution_parameter=leiden_resolution)
        
        cluster_labels = list(partition.membership)
        
        print(f"Número de clusters encontrados: {len(set(cluster_labels))}")
        
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
        
        print("El proceso ha terminado exitosamente!")
        return self.results

    def plot_clusters(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 10)):
        """Visualizar los clusters encontrados"""
        if self.results is None:
            print("Ejecutar detect_dynamic_clusters() primero")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Network plot
        if self.network and len(self.network.vs) > 0:
            try:
                pos = self.network.layout('fr')
                colors = plt.cm.Set3(np.array(self.results['partition'].membership))

                # Crear el plot de red manualmente ya que ig.plot puede tener problemas
                edge_list = [(e.source, e.target) for e in self.network.es]
                pos_dict = {i: pos[i] for i in range(len(pos))}

                # Dibujar edges
                for edge in edge_list:
                    x_coords = [pos_dict[edge[0]][0], pos_dict[edge[1]][0]]
                    y_coords = [pos_dict[edge[0]][1], pos_dict[edge[1]][1]]
                    axes[0, 0].plot(x_coords, y_coords, 'k-', alpha=0.3, linewidth=0.5)

                # Dibujar nodos
                x_pos = [pos_dict[i][0] for i in range(len(pos_dict))]
                y_pos = [pos_dict[i][1] for i in range(len(pos_dict))]
                scatter = axes[0, 0].scatter(x_pos, y_pos, c=colors, s=50, alpha=0.8)

                # Agregar nombres de residuos como etiquetas
                residue_names = self.network.vs['name']
                for i, (x, y) in enumerate(zip(x_pos, y_pos)):
                    axes[0, 0].annotate(residue_names[i], (x, y),
                                        xytext=(5, 5), textcoords='offset points',
                                        fontsize=8, alpha=0.7,
                                        bbox=dict(boxstyle='round,pad=0.3',
                                                  facecolor='white', alpha=0.7, edgecolor='none'))

                axes[0, 0].set_title('Red de Residuos Clusterizados')
                axes[0, 0].set_xlabel('Coordenada X')
                axes[0, 0].set_ylabel('Coordenada Y')

            except Exception as e:
                axes[0, 0].text(0.5, 0.5, f'Error en visualización de red:\n{str(e)}',
                                transform=axes[0, 0].transAxes, ha='center', va='center')
                axes[0, 0].set_title('Red de Residuos (Error)')

        # 2. Cluster distribution
        if self.clusters is not None and len(self.clusters) > 0:
            cluster_counts = self.clusters['Cluster'].value_counts().sort_index()
            axes[0, 1].bar(range(len(cluster_counts)), cluster_counts.values,
                           color=plt.cm.Set3(np.arange(len(cluster_counts))))
            axes[0, 1].set_xlabel('Cluster ID')
            axes[0, 1].set_ylabel('Número de Residuos')
            axes[0, 1].set_title('Distribución de Clusters')
            axes[0, 1].set_xticks(range(len(cluster_counts)))
            axes[0, 1].set_xticklabels(cluster_counts.index)

        # 3. Frustration statistics
        if 'statistics' in self.results:
            stats = self.results['statistics']
            selected = stats['Selected']
            axes[1, 0].scatter(stats['Mean'][~selected], stats['FrstRange'][~selected],
                               c='lightgray', alpha=0.6, label='No seleccionados', s=30)
            axes[1, 0].scatter(stats['Mean'][selected], stats['FrstRange'][selected],
                               c='red', alpha=0.8, label='Seleccionados', s=30)
            axes[1, 0].set_xlabel('Media de Frustración')
            axes[1, 0].set_ylabel('Rango de Frustración')
            axes[1, 0].set_title('Estadísticas de Frustración')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 4. PCA explained variance
        if 'pca_model' in self.results:
            pca_var = self.results['pca_model'].explained_variance_ratio_
            axes[1, 1].bar(range(1, len(pca_var) + 1), pca_var,
                           color='steelblue', alpha=0.7)
            axes[1, 1].set_xlabel('Componente Principal')
            axes[1, 1].set_ylabel('Varianza Explicada')
            axes[1, 1].set_title('Varianza Explicada por PCA')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {save_path}")

        plt.show()
        return fig
    
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
            print("Ejecutar detect_dynamic_clusters() primero")
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
            print("No hay resultados para guardar. Ejecutar detect_dynamic_clusters() primero.")
            return
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar clusters
        clusters_file = os.path.join(output_dir, "clusters.csv")
        if self.clusters is not None:
            self.clusters.to_csv(clusters_file, index=False)
            print(f"Información de clusters guardada en: {clusters_file}")
        
        # Guardar estadísticas
        stats_file = os.path.join(output_dir, "statistics.csv")
        if 'statistics' in self.results:
            self.results['statistics'].to_csv(stats_file, index=False)
            print(f"Estadísticas guardadas en: {stats_file}")
        
        # Guardar parámetros
        params_file = os.path.join(output_dir, "parameters.txt")
        if 'parameters' in self.results:
            with open(params_file, 'w') as f:
                f.write("Parámetros utilizados:\n")
                f.write("=====================\n\n")
                for key, value in self.results['parameters'].items():
                    f.write(f"{key}: {value}\n")
            print(f"Parámetros guardados en: {params_file}")
        
        # Guardar gráficos
        plot_file = os.path.join(output_dir, "clusters_plot.png")
        self.plot_clusters(save_path=plot_file)
        
        # Resumen
        summary_file = os.path.join(output_dir, "summary.txt")
        with open(summary_file, 'w') as f:
            f.write("RESUMEN DE ANÁLISIS DE CLUSTERS DINÁMICOS\n")
            f.write("=========================================\n\n")
            
            if 'statistics' in self.results:
                total_residues = len(self.results['statistics'])
                selected_residues = np.sum(self.results['statistics']['Selected'])
                f.write(f"Total de residuos: {total_residues}\n")
                f.write(f"Residuos seleccionados: {selected_residues}\n")
            
            if self.clusters is not None:
                n_clusters = self.clusters['Cluster'].nunique()
                f.write(f"Número de clusters: {n_clusters}\n\n")
                
                # Resumen por cluster
                f.write("Resumen por cluster:\n")
                f.write("-------------------\n")
                cluster_summary = self.clusters.groupby('Cluster').agg({
                    'Residue': 'count',
                    'Chain': lambda x: ', '.join(x.astype(str).unique()),
                    'Mean': ['mean', 'std'],
                    'FrstRange': ['mean', 'std']
                }).round(3)
                f.write(cluster_summary.to_string())
        
        print(f"Resumen guardado en: {summary_file}")
        print(f"\nTodos los resultados guardados en el directorio: {output_dir}")


def main():
    """Función principal para ejecutar desde línea de comandos"""
    
    parser = argparse.ArgumentParser(
        description="Análisis de clusters dinámicos de frustración",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python cluster.py datos.csv
  python cluster.py datos.xlsx --chain_col cadena --residue_col residuo
  python cluster.py datos.csv --min_corr 0.8 --output results_folder
  python cluster.py datos.csv --help-params
        """
    )
    
    # Argumento requerido
    parser.add_argument('file', help='Ruta al archivo de datos (CSV o Excel)')
    
    # Argumentos de columnas
    parser.add_argument('--chain_col', default='chain', 
                       help='Nombre de la columna de cadenas (default: chain)')
    parser.add_argument('--residue_col', default='residue',
                       help='Nombre de la columna de residuos (default: residue)')
    
    # Parámetros del algoritmo
    parser.add_argument('--loess_span', type=float, default=0.05,
                       help='Parámetro de suavizado LOESS (default: 0.05)')
    parser.add_argument('--min_frst_range', type=float, default=0.7,
                       help='Percentil mínimo del rango de frustración (default: 0.7)')
    parser.add_argument('--filt_mean', type=float, default=0.15,
                       help='Umbral de filtrado por media (default: 0.15)')
    parser.add_argument('--n_components', type=int, default=10,
                       help='Número de componentes PCA (default: 10)')
    parser.add_argument('--min_corr', type=float, default=0.95,
                       help='Correlación mínima para conexiones (default: 0.95)')
    parser.add_argument('--leiden_resolution', type=float, default=1.0,
                       help='Resolución para clustering Leiden (default: 1.0)')
    parser.add_argument('--corr_type', choices=['spearman', 'pearson'], default='spearman',
                       help='Tipo de correlación (default: spearman)')
    
    # Argumentos de salida
    parser.add_argument('--output', '-o', default='cluster_results',
                       help='Directorio de salida (default: cluster_results)')
    parser.add_argument('--no_plot', action='store_true',
                       help='No mostrar gráficos interactivos')
    
    # Ayuda adicional
    parser.add_argument('--help-params', action='store_true',
                       help='Mostrar explicación detallada de parámetros')
    
    args = parser.parse_args()
    
    if args.help_params:
        print("""
EXPLICACIÓN DETALLADA DE PARÁMETROS:
===================================

--loess_span (0.01-0.2): 
    Control de suavizado. Valores más bajos = menos suavizado.
    
--min_frst_range (0.5-0.9):
    Percentil para filtrar por rango dinámico. 0.7 = solo residuos con rango 
    en el top 30% más variable.
    
--filt_mean (0.05-0.3):
    Umbral absoluto para filtrar por media de frustración. Solo residuos con
    |media| > umbral.
    
--n_components (5-20):
    Dimensiones PCA. Más componentes = más información pero más ruido.
    
--min_corr (0.7-0.99):
    Correlación mínima entre residuos para formar conexiones. Valores altos
    = clusters más estrictos.
    
--leiden_resolution (0.1-2.0):
    Control del tamaño de clusters. Valores altos = más clusters pequeños.
    
CONSEJOS:
- Para datos con poco ruido: min_corr alto (>0.9)
- Para encontrar más clusters: leiden_resolution alto (>1.5)
- Para datos muy variables: filt_mean bajo (<0.1)
        """)
        return
    
    # Verificar que el archivo existe
    if not os.path.exists(args.file):
        print(f"Error: El archivo '{args.file}' no existe.")
        sys.exit(1)
    
    print(f"Analizando archivo: {args.file}")
    print("="*50)
    
    try:
        # Cargar datos
        if args.file.endswith('.csv'):
            df = pd.read_csv(args.file)
        elif args.file.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(args.file)
        else:
            print("Error: Formato de archivo no soportado. Use CSV o Excel.")
            sys.exit(1)
        
        print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f"Columnas detectadas: {list(df.columns)}")
        
        # Verificar columnas requeridas
        if args.chain_col not in df.columns:
            print(f"Error: Columna '{args.chain_col}' no encontrada en los datos.")
            sys.exit(1)
        
        if args.residue_col not in df.columns:
            print(f"Error: Columna '{args.residue_col}' no encontrada en los datos.")
            sys.exit(1)
        
        # Verificar columnas de frames
        frame_cols = [col for col in df.columns if col.startswith('frame')]
        if len(frame_cols) == 0:
            print("Error: No se encontraron columnas que empiecen con 'frame'.")
            sys.exit(1)
        
        print(f"Columnas de frames detectadas: {len(frame_cols)}")
        
        # Crear analizador
        analyzer = DynamicClustersAnalyzer()
        
        # Ejecutar análisis
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
            print("Error: No se pudieron generar resultados. Revisar parámetros.")
            sys.exit(1)
        
        # Mostrar resumen
        print("\n" + "="*50)
        print("RESUMEN DE RESULTADOS")
        print("="*50)
        
        if analyzer.clusters is not None:
            n_clusters = analyzer.clusters['Cluster'].nunique()
            print(f"Clusters encontrados: {n_clusters}")
            
            cluster_summary = analyzer.clusters.groupby('Cluster').agg({
                'Residue': 'count',
                'Chain': lambda x: ', '.join(x.astype(str).unique()),
                'Mean': 'mean',
                'FrstRange': 'mean'
            }).round(3)
            print("\nResumen por cluster:")
            print(cluster_summary)
        
        # Guardar resultados
        analyzer.save_results(args.output)
        
        # Mostrar gráficos si se solicita
        if not args.no_plot:
            print("\nMostrando gráficos...")
            analyzer.plot_clusters()
        
        print(f"\n✓ Análisis completado exitosamente!")
        print(f"✓ Resultados guardados en: {args.output}")
        
    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
