import pandas as pd
import os
import re
from frustramotion.analysis.single import SingleResidueAnalyzer

def generate_chimerax_script(df, pdb_file, output_path, metric='hotspots', chain_id=None):
    """
    Generates a ChimeraX (.cxc) script to map FrustraMotion metrics onto a 3D structure.
    """
    # Ensure it's a Single Residue DF
    if 'ResID1' in df.columns or 'Residue' not in df.columns:
        print("[!] Error: Expected a Single Residue DataFrame, but a Contact Network DataFrame was provided.")
        print("    Please use a single residue CSV for this specific export function.")
        return False

    if chain_id is None:
        chain_id = df['Chain'].iloc[0]
        
    chain_data = df[df['Chain'] == chain_id].copy()
    
    if chain_data.empty:
        print(f"[!] Error: No data found for Chain {chain_id}")
        return False

    traj = SingleResidueAnalyzer(chain_data)
    
    # Calculate Metric
    metric_title = ""
    if metric == 'hotspots':
        stats = traj.get_hotspots(top_n=9999).reset_index()
        value_col = 'Dynamic_Score'
        metric_title = "Dynamic Variance"
    elif metric == 'entropy':
        stats = traj.get_state_entropy().reset_index()
        value_col = 'Shannon_Entropy'
        metric_title = "State Entropy"
    elif metric == 'flipping':
        stats = traj.get_flipping_rate().reset_index()
        value_col = 'Flipping_Rate'
        metric_title = "Flipping Rate"
    else:
        print(f"[!] Error: Metric '{metric}' not supported.")
        return False

    def get_resnum(res_str):
        match = re.search(r'\d+', str(res_str))
        return match.group() if match else ""
    
    stats['ResNum'] = stats['Residue'].apply(get_resnum)
    
    # Normalize 0-100 for coloring
    min_val = stats[value_col].min()
    max_val = stats[value_col].max()
    stats['Bfactor'] = ((stats[value_col] - min_val) / (max_val - min_val)) * 100

    abs_pdb_path = os.path.abspath(pdb_file).replace('\\', '/')

    with open(output_path, 'w') as cxc:
        cxc.write(f"# FrustraMotion ChimeraX Export | Metric: {metric_title}\n\n")
        
        # Base Environment
        cxc.write("set bgColor white\n")
        cxc.write("lighting soft\n\n")
        
        # Load Model
        cxc.write(f"open \"{abs_pdb_path}\"\n")
        cxc.write("hide atoms\n")
        cxc.write(f"show /{chain_id} cartoon\n\n")
        
        # Inject custom b-factor values per residue
        cxc.write("# Injecting analytical values into B-factor...\n")
        for _, row in stats.iterrows():
            resnum = row['ResNum']
            bfactor = row['Bfactor']
            # ChimeraX command to set attribute: setattr target attribute value selection
            cxc.write(f"setattr /{chain_id}:{resnum} atoms bfactor {bfactor:.2f}\n")
            
        # Coloring command (Blue -> White -> Red)
        cxc.write("\n# Applying Heatmap Colors\n")
        cxc.write(f"color byattribute bfactor /{chain_id} palette blue:white:red\n")
            
        cxc.write("\nview\n")
        
    print(f" -> ChimeraX 3D Script generated successfully: {output_path}")
    return True