import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import argparse

def load_frustration_data(csv_file):
    """Carga los datos del CSV de frustración"""
    df = pd.read_csv(csv_file)
    return df

def create_contact_matrix(df, value_column='FrstIndex', aggregation='max_abs'):
    """
    Crea una matriz de contactos entre residuos
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame con los datos de contactos
    value_column : str
        Columna a usar para los valores ('FrstIndex' o 'FrstState')
    aggregation : str
        Método de agregación: 'max_abs', 'mean', 'first', 'last'
    """
    # Obtener lista única de residuos
    residues = sorted(set(df['Res1'].unique()) | set(df['Res2'].unique()))
    n_res = len(residues)
    
    # Crear mapeo de residuo a índice
    res_to_idx = {res: idx for idx, res in enumerate(residues)}
    
    # Inicializar matriz
    if value_column == 'FrstIndex':
        matrix = np.full((n_res, n_res), np.nan)
    else:
        matrix = np.full((n_res, n_res), '', dtype=object)
    
    # Llenar la matriz
    for _, row in df.iterrows():
        i = res_to_idx[row['Res1']]
        j = res_to_idx[row['Res2']]
        
        if value_column == 'FrstIndex':
            value = float(row[value_column])
            
            if aggregation == 'max_abs':
                if np.isnan(matrix[i, j]) or abs(value) > abs(matrix[i, j]):
                    matrix[i, j] = value
                    matrix[j, i] = value
            elif aggregation == 'mean':
                if np.isnan(matrix[i, j]):
                    matrix[i, j] = value
                    matrix[j, i] = value
                else:
                    matrix[i, j] = (matrix[i, j] + value) / 2
                    matrix[j, i] = matrix[i, j]
            elif aggregation == 'first':
                if np.isnan(matrix[i, j]):
                    matrix[i, j] = value
                    matrix[j, i] = value
        else:
            matrix[i, j] = row[value_column]
            matrix[j, i] = row[value_column]
    
    return matrix, residues

def plot_frustration_heatmap(matrix, residues, title='Mapa de Contactos de Frustración',
                            cmap_type='diverging', vmin=None, vmax=None, 
                            figsize=(12, 10), save_path=None, dpi=300):
    """
    Crea un mapa de calor de la matriz de contactos
    
    Parameters:
    -----------
    matrix : numpy array
        Matriz de contactos
    residues : list
        Lista de números de residuos
    title : str
        Título del gráfico
    cmap_type : str
        Tipo de colormap: 'diverging', 'sequential', 'custom'
    figsize : tuple
        Tamaño de la figura
    save_path : str
        Ruta para guardar la figura (None para solo mostrar)
    dpi : int
        Resolución para guardar
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Definir colormap
    if cmap_type == 'diverging':
        # Azul (frustrado) - Blanco (neutro) - Rojo (favorable)
        cmap = LinearSegmentedColormap.from_list('frustration',
                                                 ['#2563eb', '#93c5fd', '#f0f0f0', 
                                                  '#fecaca', '#dc2626'])
    elif cmap_type == 'sequential':
        cmap = 'viridis'
    elif cmap_type == 'custom':
        # Azul fuerte para muy frustrado, rojo para muy favorable
        cmap = LinearSegmentedColormap.from_list('custom_frst',
                                                 ['#1e3a8a', '#3b82f6', '#e5e7eb', 
                                                  '#f97316', '#dc2626'])
    
    # Calcular límites si no se proporcionan
    if vmin is None:
        vmin = np.nanmin(matrix)
    if vmax is None:
        vmax = np.nanmax(matrix)
    
    # Crear heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', 
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Configurar ejes
    ax.set_xticks(range(len(residues)))
    ax.set_yticks(range(len(residues)))
    ax.set_xticklabels(residues, rotation=90, fontsize=8)
    ax.set_yticklabels(residues, fontsize=8)
    
    # Etiquetas
    ax.set_xlabel('Número de Residuo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de Residuo', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Frustration Index', rotation=270, labelpad=20, 
                   fontsize=11, fontweight='bold')
    
    # Grid sutil
    ax.set_xticks(np.arange(len(residues))-0.5, minor=True)
    ax.set_yticks(np.arange(len(residues))-0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()
    return fig, ax

def plot_categorical_heatmap(matrix, residues, title='Mapa de Estados de Frustración',
                            figsize=(12, 10), save_path=None, dpi=300):
    """
    Crea un mapa de calor categórico basado en FrstState
    """
    # Mapeo de estados a valores numéricos
    state_map = {
        'highly': 2,
        'neutral': 1,
        'minimally': 0,
        '': np.nan
    }
    
    # Convertir matriz de strings a numérica
    numeric_matrix = np.vectorize(lambda x: state_map.get(x, np.nan))(matrix)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colormap personalizado para estados
    colors = ['#22c55e', '#eab308', '#ef4444', '#e5e7eb']  # verde, amarillo, rojo, gris
    cmap = LinearSegmentedColormap.from_list('states', colors, N=4)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(numeric_matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    
    # Configurar ejes
    ax.set_xticks(range(len(residues)))
    ax.set_yticks(range(len(residues)))
    ax.set_xticklabels(residues, rotation=90, fontsize=8)
    ax.set_yticklabels(residues, fontsize=8)
    
    ax.set_xlabel('Número de Residuo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de Residuo', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar con etiquetas categóricas
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, 
                       boundaries=bounds, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Minimally', 'Neutral', 'Highly'])
    cbar.set_label('Frustration State', rotation=270, labelpad=20, 
                   fontsize=11, fontweight='bold')
    
    # Grid
    ax.set_xticks(np.arange(len(residues))-0.5, minor=True)
    ax.set_yticks(np.arange(len(residues))-0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()
    return fig, ax

def plot_statistics(df, save_path=None, dpi=300):
    """Crea gráficos de estadísticas descriptivas"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribución de FrstIndex
    ax = axes[0, 0]
    df['FrstIndex'].hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.set_xlabel('Frustration Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax.set_title('Distribución de Frustration Index', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Conteo por Welltype
    ax = axes[0, 1]
    welltype_counts = df['Welltype'].value_counts()
    welltype_counts.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
    ax.set_xlabel('Well Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Número de Contactos', fontsize=11, fontweight='bold')
    ax.set_title('Distribución de Tipos de Contacto', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    
    # 3. Conteo por FrstState
    ax = axes[1, 0]
    state_counts = df['FrstState'].value_counts()
    colors_state = {'highly': '#ef4444', 'neutral': '#eab308', 'minimally': '#22c55e'}
    state_colors = [colors_state.get(state, 'gray') for state in state_counts.index]
    state_counts.plot(kind='bar', ax=ax, color=state_colors, edgecolor='black')
    ax.set_xlabel('Frustration State', fontsize=11, fontweight='bold')
    ax.set_ylabel('Número de Contactos', fontsize=11, fontweight='bold')
    ax.set_title('Distribución de Estados de Frustración', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    
    # 4. FrstIndex por Welltype (boxplot)
    ax = axes[1, 1]
    df.boxplot(column='FrstIndex', by='Welltype', ax=ax, patch_artist=True)
    ax.set_xlabel('Well Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frustration Index', fontsize=11, fontweight='bold')
    ax.set_title('Frustration Index por Tipo de Contacto', fontsize=12, fontweight='bold')
    ax.get_figure().suptitle('')  # Remover título automático
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Estadísticas guardadas en: {save_path}")
    
    plt.show()
    return fig

def main():
    parser = argparse.ArgumentParser(
        description='Visualización de mapa de contactos de frustración para papers científicos'
    )
    parser.add_argument('input_file', type=str, help='Archivo CSV con datos de frustración')
    parser.add_argument('--output-prefix', type=str, default='frustration',
                       help='Prefijo para archivos de salida (default: frustration)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Resolución de las figuras (default: 300)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[12, 10],
                       help='Tamaño de figura (width height) (default: 12 10)')
    parser.add_argument('--filter-welltype', type=str, default=None,
                       help='Filtrar por tipo de well (e.g., "short")')
    parser.add_argument('--filter-state', type=str, default=None,
                       help='Filtrar por estado (e.g., "highly")')
    parser.add_argument('--cmap', type=str, default='diverging',
                       choices=['diverging', 'sequential', 'custom'],
                       help='Tipo de colormap (default: diverging)')
    parser.add_argument('--no-stats', action='store_true',
                       help='No generar gráficos de estadísticas')
    parser.add_argument('--no-categorical', action='store_true',
                       help='No generar mapa categórico')
    
    args = parser.parse_args()
    
    # Cargar datos
    print(f"Cargando datos desde: {args.input_file}")
    df = load_frustration_data(args.input_file)
    print(f"Datos cargados: {len(df)} contactos")
    
    # Aplicar filtros si se especifican
    if args.filter_welltype:
        df = df[df['Welltype'] == args.filter_welltype]
        print(f"Filtrado por Welltype='{args.filter_welltype}': {len(df)} contactos")
    
    if args.filter_state:
        df = df[df['FrstState'] == args.filter_state]
        print(f"Filtrado por FrstState='{args.filter_state}': {len(df)} contactos")
    
    # Crear matriz de contactos (FrstIndex)
    print("\nCreando matriz de contactos por FrstIndex...")
    matrix_frst, residues = create_contact_matrix(df, value_column='FrstIndex')
    
    # Generar título descriptivo
    title_parts = ['Mapa de Contactos de Frustración']
    if args.filter_welltype:
        title_parts.append(f'({args.filter_welltype})')
    if args.filter_state:
        title_parts.append(f'[{args.filter_state}]')
    title = ' '.join(title_parts)
    
    # Plot mapa de calor por FrstIndex
    output_path = f"{args.output_prefix}_heatmap_frstindex.png"
    plot_frustration_heatmap(
        matrix_frst, residues, 
        title=title,
        cmap_type=args.cmap,
        figsize=tuple(args.figsize),
        save_path=output_path,
        dpi=args.dpi
    )
    
    # Plot mapa categórico por FrstState
    if not args.no_categorical:
        print("\nCreando matriz de contactos por FrstState...")
        matrix_state, _ = create_contact_matrix(df, value_column='FrstState')
        output_path = f"{args.output_prefix}_heatmap_categorical.png"
        plot_categorical_heatmap(
            matrix_state, residues,
            title=title.replace('Frustración', 'Estados de Frustración'),
            figsize=tuple(args.figsize),
            save_path=output_path,
            dpi=args.dpi
        )
    
    # Plot estadísticas
    if not args.no_stats:
        print("\nGenerando gráficos de estadísticas...")
        output_path = f"{args.output_prefix}_statistics.png"
        plot_statistics(df, save_path=output_path, dpi=args.dpi)
    
    print("\n¡Visualización completada!")
    print(f"\nResumen:")
    print(f"  - Contactos totales: {len(df)}")
    print(f"  - Residuos únicos: {len(residues)}")
    print(f"  - Rango FrstIndex: [{df['FrstIndex'].min():.3f}, {df['FrstIndex'].max():.3f}]")
    print(f"  - Media FrstIndex: {df['FrstIndex'].mean():.3f}")

if __name__ == "__main__":
    main()