import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats
from matplotlib.gridspec import GridSpec

def load_chain_data(data_dir, chains=['A', 'B', 'C', 'D', 'E']):
    """
    Carga los datos de todas las cadenas
    
    Parameters:
    -----------
    data_dir : str
        Directorio con los archivos CSV
    chains : list
        Lista de cadenas a cargar
    
    Returns:
    --------
    dict : Diccionario con DataFrames por cadena
    """
    chain_data = {}
    ascii_codes = {'A': 65, 'B': 66, 'C': 67, 'D': 68, 'E': 69}
    
    for chain in chains:
        ascii_code = ascii_codes[chain]
        filepath = Path(data_dir) / f"frustration_chain_{chain}_ascii{ascii_code}.csv"
        
        if filepath.exists():
            df = pd.read_csv(filepath)
            chain_data[chain] = df
            print(f"Cargada cadena {chain}: {len(df)} residuos")
        else:
            print(f"ADVERTENCIA: No se encontró archivo para cadena {chain}: {filepath}")
    
    return chain_data

def extract_residue_timeseries(chain_data, residue_num):
    """
    Extrae la serie temporal de un residuo específico de todas las cadenas
    
    Parameters:
    -----------
    chain_data : dict
        Diccionario con DataFrames por cadena
    residue_num : int
        Número de residuo a extraer
    
    Returns:
    --------
    DataFrame : DataFrame con columnas [frame, chain_A, chain_B, ...]
    """
    timeseries = {}
    
    for chain, df in chain_data.items():
        # Buscar el residuo en el DataFrame
        residue_row = df[df['residue'].str.contains(f'{residue_num}', na=False)]
        
        if len(residue_row) > 0:
            # Extraer solo las columnas de frames
            frame_cols = [col for col in df.columns if col.startswith('frame')]
            values = residue_row[frame_cols].values.flatten()
            timeseries[f'chain_{chain}'] = values
        else:
            print(f"ADVERTENCIA: Residuo {residue_num} no encontrado en cadena {chain}")
    
    # Crear DataFrame
    if timeseries:
        # Obtener números de frame de las columnas
        frame_cols = [col for col in chain_data[list(chain_data.keys())[0]].columns if col.startswith('frame')]
        frames = [int(col.replace('frame', '')) for col in frame_cols]
        
        df_ts = pd.DataFrame(timeseries)
        df_ts.insert(0, 'frame', frames)
        return df_ts
    else:
        return None

def plot_multi_chain_timeseries(residues_data, residue_names, chain_colors=None,
                                y_top=2, y_bottom=-4, save_path=None, dpi=300):
    """
    Plot de series temporales para múltiples residuos y cadenas

    Parameters:
    -----------
    residues_data : dict
        Diccionario {residue_num: DataFrame_timeseries}
    residue_names : dict
        Diccionario {residue_num: 'Y188', 'H187', etc}
    chain_colors : dict
        Diccionario opcional {'chain_A': '#color', ...}
    y_top : float
        Valor superior del eje Y (default 2)
    y_bottom : float
        Valor inferior del eje Y (default -4)
    """

    n_residues = len(residues_data)
    fig, axes = plt.subplots(n_residues, 1, figsize=(16, 5*n_residues), sharex=True)

    if n_residues == 1:
        axes = [axes]

    fixed_colors = {
        'chain_A': "#0d0887",  # violeta profundo
        'chain_B': "#5b02a3",  # púrpura
        'chain_C': "#9a179b",  # magenta/fucsia
        'chain_D': "#e16462",  # naranja rojizo
        'chain_E': "#fcca30"   # amarillo cálido
    }

    fallback_palette = sns.color_palette("husl", 6).as_hex()

    for idx, (res_num, df_ts) in enumerate(residues_data.items()):
        ax = axes[idx]

        chain_cols = [col for col in df_ts.columns if col.startswith('chain_')]

        for chain_col in chain_cols:
            chain_letter = chain_col.split('_')[1]

            if chain_col in fixed_colors:
                color = fixed_colors[chain_col]
            else:
                color = fallback_palette[len(fixed_colors) % len(fallback_palette)]

            ax.plot(
                df_ts['frame'], df_ts[chain_col],
                label=f'Cadena {chain_letter}',
                color=color,
                linewidth=2, alpha=0.8
            )
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Eje Y invertido y rango fijo
        ax.set_ylim(2, -4)

        # Estética
        ax.set_ylabel('Frustration Index', fontsize=12, fontweight='bold')
        ax.set_title(
            f'{residue_names[res_num]} - Evolución Temporal',
            fontsize=14, fontweight='bold'
        )
        ax.legend(loc='upper right', ncol=5, framealpha=0.9)
        ax.grid(alpha=0.3)

        threshold = -0.5
        mean_vals = df_ts[chain_cols].mean(axis=1)
        all_frustrated = mean_vals < threshold

        if all_frustrated.any():
            ax.fill_between(
                df_ts['frame'],
                2, -4,
                where=all_frustrated,
                alpha=0.1, color='red'
            )

    axes[-1].set_xlabel('Frame', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")

    plt.show()
    return fig


def plot_correlation_heatmap(residues_data, residue_names, save_path=None, dpi=300):
    """
    Heatmap de correlación entre cadenas para cada residuo
    """
    n_residues = len(residues_data)
    fig, axes = plt.subplots(1, n_residues, figsize=(6*n_residues, 5))
    
    if n_residues == 1:
        axes = [axes]
    
    for idx, (res_num, df_ts) in enumerate(residues_data.items()):
        ax = axes[idx]
        
        # Extraer solo las columnas de cadenas
        chain_cols = [col for col in df_ts.columns if col.startswith('chain_')]
        corr_matrix = df_ts[chain_cols].corr()
        
        # Renombrar para mejor visualización
        corr_matrix.index = [col.split('_')[1] for col in corr_matrix.index]
        corr_matrix.columns = [col.split('_')[1] for col in corr_matrix.columns]
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Añadir valores en las celdas
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", 
                             color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white",
                             fontsize=11, fontweight='bold')
        
        # Styling
        ax.set_xticks(range(len(corr_matrix)))
        ax.set_yticks(range(len(corr_matrix)))
        ax.set_xticklabels(corr_matrix.columns, fontsize=11)
        ax.set_yticklabels(corr_matrix.index, fontsize=11)
        ax.set_title(f'{residue_names[res_num]}\nCorrelación Inter-Cadena', 
                    fontsize=13, fontweight='bold')
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Heatmap de correlación guardado en: {save_path}")
    
    plt.show()
    return fig

def generate_cooperativity_report(residues_data, residue_names, output_path='cooperativity_report.txt'):
    """
    Genera un reporte de texto con estadísticas de cooperatividad
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE COOPERATIVIDAD INTER-CADENA\n")
        f.write("="*80 + "\n\n")
        
        for res_num, df_ts in residues_data.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"RESIDUO: {residue_names[res_num]}\n")
            f.write(f"{'='*80}\n\n")
            
            chain_cols = [col for col in df_ts.columns if col.startswith('chain_')]
            
            # Estadísticas básicas por cadena
            f.write("1. ESTADÍSTICAS BÁSICAS POR CADENA\n")
            f.write("-" * 80 + "\n")
            
            for chain_col in chain_cols:
                chain_letter = chain_col.split('_')[1]
                values = df_ts[chain_col].values
                
                f.write(f"\nCadena {chain_letter}:\n")
                f.write(f"  Media: {np.mean(values):.4f}\n")
                f.write(f"  Desviación estándar: {np.std(values):.4f}\n")
                f.write(f"  Mínimo: {np.min(values):.4f}\n")
                f.write(f"  Máximo: {np.max(values):.4f}\n")
                f.write(f"  % frames frustrados (< -0.5): {(values < -0.5).sum() / len(values) * 100:.1f}%\n")
            
            # Correlaciones
            f.write("\n\n2. MATRIZ DE CORRELACIÓN\n")
            f.write("-" * 80 + "\n")
            corr_matrix = df_ts[chain_cols].corr()
            corr_matrix.index = [col.split('_')[1] for col in corr_matrix.index]
            corr_matrix.columns = [col.split('_')[1] for col in corr_matrix.columns]
            f.write(corr_matrix.to_string())
            f.write("\n")
            
            # Correlaciones más altas
            f.write("\n\n3. PARES DE CADENAS MÁS CORRELACIONADAS\n")
            f.write("-" * 80 + "\n")
            
            correlations = []
            for i, chain_i in enumerate(chain_cols):
                for j, chain_j in enumerate(chain_cols):
                    if i < j:
                        corr = np.corrcoef(df_ts[chain_i].values, df_ts[chain_j].values)[0, 1]
                        correlations.append((chain_i.split('_')[1], chain_j.split('_')[1], corr))
            
            correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            for chain_i, chain_j, corr in correlations:
                f.write(f"Cadenas {chain_i}-{chain_j}: {corr:7.4f}\n")
            
            # Eventos cooperativos
            f.write("\n\n4. EVENTOS COOPERATIVOS\n")
            f.write("-" * 80 + "\n")
            
            # Contar frames donde todas las cadenas están frustradas
            all_values = df_ts[chain_cols].values
            all_frustrated = np.all(all_values < -0.5, axis=1)
            n_cooperative_frustrated = all_frustrated.sum()
            
            f.write(f"Frames con todas las cadenas frustradas (< -0.5): {n_cooperative_frustrated} ")
            f.write(f"({n_cooperative_frustrated / len(df_ts) * 100:.1f}%)\n")
            
            # Transiciones simultáneas
            changes = df_ts[chain_cols].diff()
            same_direction_all = np.all((changes.values > 0) | (changes.values < 0), axis=1)
            same_sign = np.all(changes.values > 0, axis=1) | np.all(changes.values < 0, axis=1)
            n_sync_transitions = (same_direction_all & same_sign).sum()
            
            f.write(f"Transiciones sincrónicas (todas cambian en misma dirección): {n_sync_transitions} ")
            f.write(f"({n_sync_transitions / len(df_ts) * 100:.1f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("="*80 + "\n")
    
    print(f"Reporte de cooperatividad guardado en: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Análisis de cooperatividad inter-cadena para residuos específicos'
    )
    parser.add_argument('data_dir', type=str, 
                       help='Directorio con los archivos CSV de las cadenas')
    parser.add_argument('--residues', type=int, nargs='+', default=[187, 188],
                       help='Números de residuos a analizar (default: 187 188)')
    parser.add_argument('--residue-names', type=str, nargs='+', default=None,
                       help='Nombres de residuos (e.g., H187 Y188). Si no se proporciona, usa números')
    parser.add_argument('--chains', type=str, nargs='+', default=['A', 'B', 'C', 'D', 'E'],
                       help='Cadenas a analizar (default: A B C D E)')
    parser.add_argument('--output-prefix', type=str, default='cooperativity',
                       help='Prefijo para archivos de salida')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Resolución de las figuras (default: 300)')
    parser.add_argument('--report', action='store_true',
                       help='Generar reporte de texto con estadísticas')
    
    args = parser.parse_args()
    
    # Procesar nombres de residuos
    if args.residue_names:
        residue_names = {num: name for num, name in zip(args.residues, args.residue_names)}
    else:
        residue_names = {num: f'Residuo {num}' for num in args.residues}
    
    print("="*80)
    print("ANÁLISIS DE COOPERATIVIDAD INTER-CADENA")
    print("="*80)
    print(f"\nDirectorio de datos: {args.data_dir}")
    print(f"Residuos a analizar: {', '.join([residue_names[r] for r in args.residues])}")
    print(f"Cadenas: {', '.join(args.chains)}\n")
    
    # Cargar datos
    print("Cargando datos de cadenas...")
    chain_data = load_chain_data(args.data_dir, args.chains)
    
    if not chain_data:
        print("ERROR: No se pudieron cargar datos de ninguna cadena")
        return
    
    # Extraer series temporales para cada residuo
    print("\nExtrayendo series temporales...")
    residues_data = {}
    
    for res_num in args.residues:
        df_ts = extract_residue_timeseries(chain_data, res_num)
        if df_ts is not None:
            residues_data[res_num] = df_ts
            print(f"  {residue_names[res_num]}: {len(df_ts)} frames")
        else:
            print(f"  ADVERTENCIA: No se pudo extraer {residue_names[res_num]}")
    
    if not residues_data:
        print("ERROR: No se pudieron extraer datos de ningún residuo")
        return
    
    # Generar visualizaciones
    print("\n" + "="*80)
    print("GENERANDO VISUALIZACIONES")
    print("="*80)
    
    # 1. Series temporales
    print("\n1. Series temporales multi-cadena...")
    plot_multi_chain_timeseries(
        residues_data, residue_names,
        save_path=f"{args.output_prefix}_timeseries.png",
        dpi=args.dpi
    )
    
    # 2. Heatmap de correlación
    print("2. Heatmap de correlación inter-cadena...")
    plot_correlation_heatmap(
        residues_data, residue_names,
        save_path=f"{args.output_prefix}_correlation_heatmap.png",
        dpi=args.dpi
    )
    
    # 3. Reporte de texto
    if args.report:
        print("3. Generando reporte de cooperatividad...")
        generate_cooperativity_report(
            residues_data, residue_names,
            output_path=f"{args.output_prefix}_report.txt"
        )
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print(f"\nArchivos generados:")
    print(f"  - {args.output_prefix}_timeseries.png")
    print(f"  - {args.output_prefix}_correlation_heatmap.png")
    if args.report:
        print(f"  - {args.output_prefix}_report.txt")

if __name__ == "__main__":
    main()