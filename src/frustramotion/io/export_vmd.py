import pandas as pd
import os
import re

from frustramotion.analysis.core import FrustrationTrajectory

def generate_vmd_script(df, pdb_file, output_path, metric='hotspots', chain_id=None):
    """
    Generates a VMD TCL script to color a protein based on dynamic FrustraMotion metrics.
    Injects the calculated metrics into the B-factor (beta) column for gradient coloring.
    """
    if chain_id is None:
        chain_id = df['Chain'].iloc[0]
        
    chain_data = df[df['Chain'] == chain_id].copy()
    
    if chain_data.empty:
        print(f"[!] Error: No data found for Chain {chain_id}")
        return False


    # 1. Initialize Analytics Engine
    traj = FrustrationTrajectory(chain_data)
    
    # 2. Calculate the requested metric
    metric_name_in_vmd = ""
    if metric == 'hotspots':
        stats = traj.get_hotspots(top_n=9999).reset_index()
        value_col = 'Dynamic_Score'
        metric_name_in_vmd = "Energetic Variance (Hotspots)"
    elif metric == 'entropy':
        stats = traj.get_state_entropy().reset_index()
        value_col = 'Shannon_Entropy'
        metric_name_in_vmd = "State Entropy (Unpredictability)"
    elif metric == 'flipping':
        stats = traj.get_flipping_rate().reset_index()
        value_col = 'Flipping_Rate'
        metric_name_in_vmd = "Flipping Rate (Transitions/Frame)"
    else:
        print(f"[!] Error: Metric '{metric}' not supported for VMD export.")
        return False

    # Extract clean residue numbers for VMD mapping
    def get_resnum(res_str):
        match = re.search(r'\d+', str(res_str))
        return match.group() if match else ""
    
    stats['ResNum'] = stats['Residue'].apply(get_resnum)
    
    # Normalize values between 0 and 100 for better VMD coloring
    min_val = stats[value_col].min()
    max_val = stats[value_col].max()
    stats['Beta_Value'] = ((stats[value_col] - min_val) / (max_val - min_val)) * 100

    # 3. Write the TCL Script
    with open(output_path, 'w') as tcl:
        tcl.write("# ==========================================\n")
        tcl.write("# FrustraMotion 3D Analytics Mapping\n")
        tcl.write(f"# Metric: {metric_name_in_vmd}\n")
        tcl.write("# ==========================================\n\n")
        
        # Load molecule
        tcl.write(f"mol new {{{pdb_file}}} type pdb waitfor all\n")
        tcl.write("color Display Background white\n")
        tcl.write("display projection Orthographic\n")
        tcl.write("axes location off\n\n")
        
        # Reset all B-factors to 0
        tcl.write("# Reset B-factors\n")
        tcl.write("set all_atoms [atomselect top all]\n")
        tcl.write("$all_atoms set beta 0\n\n")
        
        # Inject the new metric values into the B-factor of each residue
        tcl.write("# Injecting FrustraMotion metrics into Beta column...\n")
        for _, row in stats.iterrows():
            resnum = row['ResNum']
            beta_val = row['Beta_Value']
            tcl.write(f"set sel [atomselect top \"chain {chain_id} and resid {resnum}\"]\n")
            tcl.write(f"$sel set beta {beta_val:.2f}\n")
            tcl.write("$sel delete\n")
            
        tcl.write("$all_atoms delete\n\n")

        # Setup the visual representation
        tcl.write("# Set up clean visual representation\n")
        tcl.write("mol delrep 0 top\n")
        
        # Base Cartoon Representation (colored by Beta gradient)
        tcl.write("mol representation NewCartoon 0.30 10.0 4.1\n")
        tcl.write("mol color Beta\n")
        tcl.write(f"mol selection \"chain {chain_id}\"\n")
        tcl.write("mol material Opaque\n")
        tcl.write("mol addrep top\n\n")
        
        # Set color scale to Blue-White-Red
        tcl.write("color scale method BWR\n")
        tcl.write("color scale midpoint 0.5\n")
        tcl.write("display resetview\n")
        
    print(f" -> VMD 3D Clean Heatmap Script generated: {output_path}")
    return True