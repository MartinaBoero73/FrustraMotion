import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import argparse
from pathlib import Path

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
    step = 5
    xticks = np.arange(0, len(residues), step)
    yticks = np.arange(0, len(residues), step)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels([residues[i] for i in xticks], rotation=90, fontsize=8)
    ax.set_yticklabels([residues[i] for i in yticks], fontsize=8)

    
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
    step = 5
    xticks = np.arange(0, len(residues), step)
    yticks = np.arange(0, len(residues), step)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels([residues[i] for i in xticks], rotation=90, fontsize=8)
    ax.set_yticklabels([residues[i] for i in yticks], fontsize=8)
    
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

def create_animated_heatmap(df, output_path='frustration_animation.gif', 
                           cmap_type='diverging', figsize=(14, 10), 
                           fps=10, dpi=100, frame_step=1):
    """
    Crea una animación del mapa de calor a través de los frames
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame completo con columna 'Frame'
    output_path : str
        Ruta para guardar la animación
    cmap_type : str
        Tipo de colormap
    figsize : tuple
        Tamaño de la figura
    fps : int
        Frames por segundo
    dpi : int
        Resolución
    frame_step : int
        Saltar cada N frames (para animaciones más rápidas)
    """
    # Obtener frames únicos
    frames = sorted(df['Frame'].unique())[::frame_step]
    print(f"Creando animación con {len(frames)} frames...")
    
    # Obtener todos los residuos únicos
    all_residues = sorted(set(df['Res1'].unique()) | set(df['Res2'].unique()))
    
    # Calcular límites globales para vmin/vmax consistentes
    vmin = df['FrstIndex'].min()
    vmax = df['FrstIndex'].max()
    
    # Definir colormap
    if cmap_type == 'diverging':
        cmap = LinearSegmentedColormap.from_list('frustration',
                                                 ['#2563eb', '#93c5fd', '#f0f0f0', 
                                                  '#fecaca', '#dc2626'])
    elif cmap_type == 'sequential':
        cmap = 'viridis'
    else:
        cmap = LinearSegmentedColormap.from_list('custom_frst',
                                                 ['#1e3a8a', '#3b82f6', '#e5e7eb', 
                                                  '#f97316', '#dc2626'])
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Inicializar imagen
    matrix, _ = create_contact_matrix(df[df['Frame'] == frames[0]], value_column='FrstIndex')
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Configurar ejes
    step = 5
    xticks = np.arange(0, len(all_residues), step)
    yticks = np.arange(0, len(all_residues), step)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels([all_residues[i] for i in xticks], rotation=90, fontsize=8)
    ax.set_yticklabels([all_residues[i] for i in yticks], fontsize=8)

    
    # Título dinámico
    title = ax.set_title(f'Frame: {frames[0]}', fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Frustration Index', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
    
    # Grid
    ax.set_xticks(np.arange(len(all_residues))-0.5, minor=True)
    ax.set_yticks(np.arange(len(all_residues))-0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.1, alpha=0.3)
    
    plt.tight_layout()
    
    def update(frame_idx):
        """Actualiza el frame de la animación"""
        current_frame = frames[frame_idx]
        df_frame = df[df['Frame'] == current_frame]
        matrix, _ = create_contact_matrix(df_frame, value_column='FrstIndex')
        
        im.set_data(matrix)
        title.set_text(f'Mapa de Frustración - Frame: {current_frame} ({frame_idx+1}/{len(frames)})')
        
        return [im, title]
    
    # Crear animación
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=True)
    
    # Guardar
    print(f"Guardando animación en {output_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close()
    
    print(f"Animación guardada exitosamente: {output_path}")
    return anim

def plot_temporal_statistics(df, save_path=None, dpi=300):
    """
    Crea gráficos de estadísticas temporales (por frame)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. FrstIndex promedio por frame
    ax = axes[0, 0]
    frame_stats = df.groupby('Frame')['FrstIndex'].agg(['mean', 'std'])
    ax.plot(frame_stats.index, frame_stats['mean'], linewidth=2, color='steelblue', label='Media')
    ax.fill_between(frame_stats.index, 
                     frame_stats['mean'] - frame_stats['std'],
                     frame_stats['mean'] + frame_stats['std'],
                     alpha=0.3, color='steelblue', label='±1 SD')
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
    ax.set_ylabel('FrstIndex Promedio', fontsize=11, fontweight='bold')
    ax.set_title('Evolución Temporal de FrstIndex', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Número de contactos por frame
    ax = axes[0, 1]
    contacts_per_frame = df.groupby('Frame').size()
    ax.plot(contacts_per_frame.index, contacts_per_frame.values, 
            linewidth=2, color='coral', marker='o', markersize=3)
    ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
    ax.set_ylabel('Número de Contactos', fontsize=11, fontweight='bold')
    ax.set_title('Contactos por Frame', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 3. Distribución de estados por frame (stacked area)
    ax = axes[0, 2]
    state_by_frame = df.groupby(['Frame', 'FrstState']).size().unstack(fill_value=0)
    state_colors = {'highly': '#ef4444', 'neutral': '#eab308', 'minimally': '#22c55e'}
    
    for state in state_by_frame.columns:
        ax.fill_between(state_by_frame.index, 0, state_by_frame[state], 
                        label=state, alpha=0.7, color=state_colors.get(state, 'gray'))
    
    ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
    ax.set_ylabel('Número de Contactos', fontsize=11, fontweight='bold')
    ax.set_title('Evolución de Estados de Frustración', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Distribución de Welltype por frame
    ax = axes[1, 0]
    welltype_by_frame = df.groupby(['Frame', 'Welltype']).size().unstack(fill_value=0)
    welltype_by_frame.plot(ax=ax, kind='area', stacked=True, alpha=0.7)
    ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
    ax.set_ylabel('Número de Contactos', fontsize=11, fontweight='bold')
    ax.set_title('Evolución de Tipos de Contacto', fontsize=12, fontweight='bold')
    ax.legend(title='Well Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    # 5. Heatmap de FrstIndex por frame (muestreado)
    ax = axes[1, 1]
    # Muestrear frames si hay demasiados
    unique_frames = sorted(df['Frame'].unique())
    if len(unique_frames) > 50:
        sample_frames = unique_frames[::len(unique_frames)//50]
    else:
        sample_frames = unique_frames
    
    pivot_data = df[df['Frame'].isin(sample_frames)].groupby('Frame')['FrstIndex'].mean()
    pivot_matrix = pivot_data.values.reshape(-1, 1)
    
    im = ax.imshow(pivot_matrix.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    ax.set_xlabel('Frame (muestreado)', fontsize=11, fontweight='bold')
    ax.set_ylabel('FrstIndex Promedio', fontsize=11, fontweight='bold')
    ax.set_title('Intensidad de Frustración vs Tiempo', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    plt.colorbar(im, ax=ax)
    
    # 6. Varianza de FrstIndex por frame
    ax = axes[1, 2]
    frame_variance = df.groupby('Frame')['FrstIndex'].var()
    ax.plot(frame_variance.index, frame_variance.values, 
            linewidth=2, color='purple', marker='o', markersize=3)
    ax.set_xlabel('Frame', fontsize=11, fontweight='bold')
    ax.set_ylabel('Varianza de FrstIndex', fontsize=11, fontweight='bold')
    ax.set_title('Variabilidad Temporal de Frustración', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Estadísticas temporales guardadas en: {save_path}")
    
    plt.show()
    return fig

def export_frame_statistics_csv(df, output_path='frame_statistics.csv'):
    """
    Exporta estadísticas por frame a CSV
    """
    # Calcular estadísticas por frame
    stats = df.groupby('Frame').agg({
        'FrstIndex': ['mean', 'std', 'min', 'max', 'median'],
        'Res1': 'count'  # Número de contactos
    }).reset_index()
    
    # Aplanar columnas multinivel
    stats.columns = ['Frame', 'FrstIndex_mean', 'FrstIndex_std', 'FrstIndex_min', 
                     'FrstIndex_max', 'FrstIndex_median', 'num_contacts']
    
    # Añadir conteos por estado
    state_counts = df.groupby(['Frame', 'FrstState']).size().unstack(fill_value=0)
    state_counts.columns = [f'count_{col}' for col in state_counts.columns]
    stats = stats.merge(state_counts, left_on='Frame', right_index=True, how='left')
    
    # Añadir conteos por welltype
    welltype_counts = df.groupby(['Frame', 'Welltype']).size().unstack(fill_value=0)
    welltype_counts.columns = [f'count_{col}' for col in welltype_counts.columns]
    stats = stats.merge(welltype_counts, left_on='Frame', right_index=True, how='left')
    
    # Guardar
    stats.to_csv(output_path, index=False)
    print(f"Estadísticas por frame exportadas a: {output_path}")
    
    return stats

def generate_text_summary(df, output_path='frustration_summary.txt'):
    """
    Genera un resumen completo en formato texto
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESUMEN DE ANÁLISIS DE CONTACTOS DE FRUSTRACIÓN\n")
        f.write("="*80 + "\n\n")
        
        # Información general
        f.write("1. INFORMACIÓN GENERAL\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total de contactos: {len(df):,}\n")
        f.write(f"Residuos únicos: {df['Res1'].nunique() + df['Res2'].nunique()}\n")
        f.write(f"Residuo mínimo: {min(df['Res1'].min(), df['Res2'].min())}\n")
        f.write(f"Residuo máximo: {max(df['Res1'].max(), df['Res2'].max())}\n")
        f.write(f"Cadenas involucradas: {', '.join(sorted(df['ChainRes1'].unique()))}\n")
        f.write("\n")
        
        # Estadísticas de Frustration Index
        f.write("2. ESTADÍSTICAS DE FRUSTRATION INDEX\n")
        f.write("-" * 80 + "\n")
        f.write(f"Media: {df['FrstIndex'].mean():.4f}\n")
        f.write(f"Desviación estándar: {df['FrstIndex'].std():.4f}\n")
        f.write(f"Mediana: {df['FrstIndex'].median():.4f}\n")
        f.write(f"Mínimo: {df['FrstIndex'].min():.4f}\n")
        f.write(f"Máximo: {df['FrstIndex'].max():.4f}\n")
        f.write(f"Cuartil 25%: {df['FrstIndex'].quantile(0.25):.4f}\n")
        f.write(f"Cuartil 75%: {df['FrstIndex'].quantile(0.75):.4f}\n")
        f.write(f"\nContactos con FrstIndex < 0 (frustrados): {(df['FrstIndex'] < 0).sum()} ({(df['FrstIndex'] < 0).sum()/len(df)*100:.1f}%)\n")
        f.write(f"Contactos con FrstIndex > 0 (favorables): {(df['FrstIndex'] > 0).sum()} ({(df['FrstIndex'] > 0).sum()/len(df)*100:.1f}%)\n")
        f.write(f"Contactos con FrstIndex = 0 (neutrales): {(df['FrstIndex'] == 0).sum()} ({(df['FrstIndex'] == 0).sum()/len(df)*100:.1f}%)\n")
        f.write("\n")
        
        # Distribución por estado de frustración
        f.write("3. DISTRIBUCIÓN POR ESTADO DE FRUSTRACIÓN\n")
        f.write("-" * 80 + "\n")
        state_counts = df['FrstState'].value_counts()
        for state, count in state_counts.items():
            percentage = count / len(df) * 100
            f.write(f"{state:20s}: {count:6d} contactos ({percentage:5.1f}%)\n")
        f.write("\n")
        
        # Distribución por tipo de contacto
        f.write("4. DISTRIBUCIÓN POR TIPO DE CONTACTO (WELLTYPE)\n")
        f.write("-" * 80 + "\n")
        welltype_counts = df['Welltype'].value_counts()
        for welltype, count in welltype_counts.items():
            percentage = count / len(df) * 100
            f.write(f"{welltype:20s}: {count:6d} contactos ({percentage:5.1f}%)\n")
        f.write("\n")
        
        # FrstIndex promedio por Welltype
        f.write("5. FRUSTRATION INDEX PROMEDIO POR TIPO DE CONTACTO\n")
        f.write("-" * 80 + "\n")
        welltype_stats = df.groupby('Welltype')['FrstIndex'].agg(['mean', 'std', 'count'])
        for welltype, row in welltype_stats.iterrows():
            f.write(f"{welltype:20s}: {row['mean']:7.4f} ± {row['std']:.4f} (n={int(row['count'])})\n")
        f.write("\n")
        
        # FrstIndex promedio por Estado
        f.write("6. FRUSTRATION INDEX PROMEDIO POR ESTADO\n")
        f.write("-" * 80 + "\n")
        state_stats = df.groupby('FrstState')['FrstIndex'].agg(['mean', 'std', 'count'])
        for state, row in state_stats.iterrows():
            f.write(f"{state:20s}: {row['mean']:7.4f} ± {row['std']:.4f} (n={int(row['count'])})\n")
        f.write("\n")
        
        # Pares de aminoácidos más frecuentes
        f.write("7. TOP 10 PARES DE AMINOÁCIDOS MÁS FRECUENTES\n")
        f.write("-" * 80 + "\n")
        aa_pairs = df.groupby(['AA1', 'AA2']).size().sort_values(ascending=False).head(10)
        for i, ((aa1, aa2), count) in enumerate(aa_pairs.items(), 1):
            percentage = count / len(df) * 100
            f.write(f"{i:2d}. {aa1}-{aa2:3s}: {count:6d} contactos ({percentage:5.1f}%)\n")
        f.write("\n")
        
        # Contactos más frustrados
        f.write("8. TOP 10 CONTACTOS MÁS FRUSTRADOS (FrstIndex más negativo)\n")
        f.write("-" * 80 + "\n")
        most_frustrated = df.nsmallest(10, 'FrstIndex')
        for i, row in enumerate(most_frustrated.itertuples(), 1):
            f.write(f"{i:2d}. {row.ResID1}-{row.ResID2}: FrstIndex={row.FrstIndex:.4f}, "
                   f"Welltype={row.Welltype}, State={row.FrstState}\n")
        f.write("\n")
        
        # Contactos más favorables
        f.write("9. TOP 10 CONTACTOS MÁS FAVORABLES (FrstIndex más positivo)\n")
        f.write("-" * 80 + "\n")
        most_favorable = df.nlargest(10, 'FrstIndex')
        for i, row in enumerate(most_favorable.itertuples(), 1):
            f.write(f"{i:2d}. {row.ResID1}-{row.ResID2}: FrstIndex={row.FrstIndex:.4f}, "
                   f"Welltype={row.Welltype}, State={row.FrstState}\n")
        f.write("\n")

        # Análisis temporal si hay frames
        if 'Frame' in df.columns:
            f.write("11. ANÁLISIS TEMPORAL\n")
            f.write("-" * 80 + "\n")
            n_frames = df['Frame'].nunique()
            f.write(f"Total de frames: {n_frames}\n")
            f.write(f"Frame inicial: {df['Frame'].min()}\n")
            f.write(f"Frame final: {df['Frame'].max()}\n")
            f.write(f"Contactos promedio por frame: {len(df)/n_frames:.2f}\n\n")
            
            # Estadísticas temporales
            frame_stats = df.groupby('Frame')['FrstIndex'].agg(['mean', 'std', 'min', 'max', 'count'])
            
            f.write("Estadísticas de FrstIndex por frame:\n")
            f.write(f"  Media temporal: {frame_stats['mean'].mean():.4f} ± {frame_stats['mean'].std():.4f}\n")
            f.write(f"  Rango de medias: [{frame_stats['mean'].min():.4f}, {frame_stats['mean'].max():.4f}]\n")
            f.write(f"  Frame con menor FrstIndex promedio: {frame_stats['mean'].idxmin()} (FrstIndex={frame_stats['mean'].min():.4f})\n")
            f.write(f"  Frame con mayor FrstIndex promedio: {frame_stats['mean'].idxmax()} (FrstIndex={frame_stats['mean'].max():.4f})\n")
            f.write(f"  Variabilidad temporal (SD de medias): {frame_stats['mean'].std():.4f}\n\n")
            
            # Evolución de estados por frame
            f.write("Evolución de estados de frustración:\n")
            state_by_frame = df.groupby(['Frame', 'FrstState']).size().unstack(fill_value=0)
            for state in state_by_frame.columns:
                mean_count = state_by_frame[state].mean()
                std_count = state_by_frame[state].std()
                f.write(f"  {state:20s}: {mean_count:.2f} ± {std_count:.2f} contactos por frame\n")
            f.write("\n")
            
            # Frames con más/menos contactos
            contacts_per_frame = df.groupby('Frame').size()
            f.write(f"Frame con más contactos: {contacts_per_frame.idxmax()} ({contacts_per_frame.max()} contactos)\n")
            f.write(f"Frame con menos contactos: {contacts_per_frame.idxmin()} ({contacts_per_frame.min()} contactos)\n")
            f.write("\n")
        
        # Residuos con más contactos
        section_num = 12 if 'Frame' in df.columns else 11
        f.write(f"{section_num}. TOP 10 RESIDUOS CON MÁS CONTACTOS\n")
        f.write("-" * 80 + "\n")
        res1_counts = df['Res1'].value_counts()
        res2_counts = df['Res2'].value_counts()
        all_res_counts = (res1_counts + res2_counts).sort_values(ascending=False).head(10)
        for i, (res, count) in enumerate(all_res_counts.items(), 1):
            # Obtener el aminoácido más común para ese residuo
            aa1 = df[df['Res1'] == res]['AA1'].mode()
            aa2 = df[df['Res2'] == res]['AA2'].mode()
            aa = aa1.iloc[0] if len(aa1) > 0 else (aa2.iloc[0] if len(aa2) > 0 else '?')
            f.write(f"{i:2d}. Residuo {int(res):4d} ({aa}): {int(count):6d} contactos\n")
        f.write("\n")
        
        # Matriz de correlación entre Welltype y FrstState
        section_num += 1
        f.write(f"{section_num}. DISTRIBUCIÓN CRUZADA: WELLTYPE vs FRSTSTATE\n")
        f.write("-" * 80 + "\n")
        cross_tab = pd.crosstab(df['Welltype'], df['FrstState'], margins=True)
        f.write(cross_tab.to_string())
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("FIN DEL RESUMEN\n")
        f.write("="*80 + "\n")
    
    print(f"Resumen en texto guardado en: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description='Visualización animada de mapa de contactos de frustración'
    )
    parser.add_argument('input_file', type=str, help='Archivo CSV con datos de frustración')
    parser.add_argument('--output-prefix', type=str, default='frustration',
                       help='Prefijo para archivos de salida (default: frustration)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Resolución de las figuras estáticas (default: 300)')
    parser.add_argument('--animation-dpi', type=int, default=100,
                       help='Resolución de la animación (default: 100, usar menos para archivos más pequeños)')
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
    parser.add_argument('--animate', action='store_true',
                       help='Crear animación por frames')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames por segundo para animación (default: 10)')
    parser.add_argument('--frame-step', type=int, default=1,
                       help='Saltar cada N frames en la animación (default: 1)')
    parser.add_argument('--temporal-stats', action='store_true',
                       help='Generar estadísticas temporales por frame')
    parser.add_argument('--export-frame-stats', action='store_true',
                       help='Exportar estadísticas por frame a CSV')
    parser.add_argument('--summary', action='store_true',
                       help='Generar resumen completo en formato texto')

    args = parser.parse_args()
    
    # Cargar datos
    print(f"Cargando datos desde: {args.input_file}")
    df = load_frustration_data(args.input_file)
    print(f"Datos cargados: {len(df)} contactos")
    
    # Verificar columna Frame
    if 'Frame' not in df.columns:
        print("ADVERTENCIA: No se encontró columna 'Frame' en los datos")
        print("Funcionalidades temporales no estarán disponibles")
        has_frames = False
    else:
        has_frames = True
        n_frames = df['Frame'].nunique()
        print(f"Frames detectados: {n_frames}")
    
    # Aplicar filtros si se especifican
    if args.filter_welltype:
        df = df[df['Welltype'] == args.filter_welltype]
        print(f"Filtrado por Welltype='{args.filter_welltype}': {len(df)} contactos")
    
    if args.filter_state:
        df = df[df['FrstState'] == args.filter_state]
        print(f"Filtrado por FrstState='{args.filter_state}': {len(df)} contactos")
    
    # Crear título descriptivo
    title_parts = ['Mapa de Contactos de Frustración']
    if args.filter_welltype:
        title_parts.append(f'({args.filter_welltype})')
    if args.filter_state:
        title_parts.append(f'[{args.filter_state}]')
    title = ' '.join(title_parts)
    
    # Estadísticas temporales
    if has_frames and args.temporal_stats:
        print("\nGenerando estadísticas temporales...")
        output_path = f"{args.output_prefix}_temporal_statistics.png"
        plot_temporal_statistics(df, save_path=output_path, dpi=args.dpi)
    
    # Exportar estadísticas por frame
    if has_frames and args.export_frame_stats:
        print("\nExportando estadísticas por frame...")
        output_path = f"{args.output_prefix}_frame_statistics.csv"
        export_frame_statistics_csv(df, output_path=output_path)
    
    # Generar resumen en texto
    if args.summary:
        print("\nGenerando resumen en texto...")
        output_path = f"{args.output_prefix}_summary.txt"
        generate_text_summary(df, output_path=output_path)
    
    # Crear animación
    if has_frames and args.animate:
        print("\nCreando animación...")
        output_path = f"{args.output_prefix}_animation.gif"
        create_animated_heatmap(
            df, 
            output_path=output_path,
            cmap_type=args.cmap,
            figsize=tuple(args.figsize),
            fps=args.fps,
            dpi=args.animation_dpi,
            frame_step=args.frame_step
        )
    
    # Crear matriz de contactos (FrstIndex) - promediado sobre todos los frames
    print("\nCreando matriz de contactos agregada por FrstIndex...")
    matrix_frst, residues = create_contact_matrix(df, value_column='FrstIndex')
    
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
    
    # Plot estadísticas generales
    if not args.no_stats:
        print("\nGenerando gráficos de estadísticas generales...")
        output_path = f"{args.output_prefix}_statistics.png"
        plot_statistics(df, save_path=output_path, dpi=args.dpi)
    
    print("\n" + "="*60)
    print("¡Visualización completada!")
    print("="*60)
    print(f"\nResumen general:")
    print(f"  - Contactos totales: {len(df)}")
    print(f"  - Residuos únicos: {len(residues)}")
    print(f"  - Rango FrstIndex: [{df['FrstIndex'].min():.3f}, {df['FrstIndex'].max():.3f}]")
    print(f"  - Media FrstIndex: {df['FrstIndex'].mean():.3f} ± {df['FrstIndex'].std():.3f}")
    
    if has_frames:
        print(f"\nResumen temporal:")
        print(f"  - Frames totales: {n_frames}")
        print(f"  - Frame inicial: {df['Frame'].min()}")
        print(f"  - Frame final: {df['Frame'].max()}")
        print(f"  - Contactos promedio por frame: {len(df)/n_frames:.1f}")

if __name__ == "__main__":
    main()