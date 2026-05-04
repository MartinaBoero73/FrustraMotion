#!/usr/bin/env python3
import pandas as pd
import sys
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize temporal frustration data from DataFrames.')
    parser.add_argument('dataframes_dir', type=str,
                      help='Directory containing the parsed frustration CSV files')
    parser.add_argument('--chain', type=str, default=None,
                      help='Specific chain to analyze (e.g., "0", "A"). If not specified, uses the first chain found.')
    parser.add_argument('--residue', type=str, required=True,
                      help='Specific residue ID to analyse (e.g., "W45" or "45"). Required.')
    parser.add_argument('--out_dir', type=str, default='./plots',
                      help='Directory where the generated plots will be saved (default: ./plots)')

    args = parser.parse_args()

    if not os.path.isdir(args.dataframes_dir):
        print(f"[!] Error: Directory {args.dataframes_dir} does not exist")
        sys.exit(1)

    # Make sure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    return args

def load_tidy_data(dataframes_dir, target_chain=None):
    """
    Loads Tidy CSV files for Single Residue frustration. 
    Intelligently loads ONLY the specific file using the compact naming convention.
    """
    if target_chain:
        chain_str = str(target_chain)[0]
        compact_id = f"_{chain_str}" if chain_str.islower() else chain_str
        
        target_suffix = f"_chain_{compact_id}.csv"
        
        for filename in os.listdir(dataframes_dir):
            if filename.endswith(target_suffix):
                filepath = os.path.join(dataframes_dir, filename)
                print(f"[*] Found file for chain '{target_chain}': {filename}")
                return pd.read_csv(filepath)
                
        print(f"[!] Error: Could not find a specific file for chain '{target_chain}' (looked for suffix '{target_suffix}')")
        sys.exit(1)
        
    print("[*] Loading all available single residue chain files...")
    all_dfs = []
    
    for filename in os.listdir(dataframes_dir):
        if filename.endswith('.csv') and '_contacts' not in filename:
            filepath = os.path.join(dataframes_dir, filename)
            df = pd.read_csv(filepath)
            all_dfs.append(df)
                
    if not all_dfs:
        print(f"[!] Error: No valid single residue CSV files found in {dataframes_dir}")
        sys.exit(1)
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    return master_df

def plot_frustration_vs_frames(df, residue_id, chain_id, out_dir):
    """
    Plot frustration values across frames for a specific residue and chain.
    """
    # Try to match the exact string (e.g., 'W45') or just the number ('45')
    res_data = df[(df['Chain'] == chain_id) & 
                  ((df['Residue'] == residue_id) | (df['Residue'].str.contains(f"^[A-Z]{residue_id}$")))]
                  
    if res_data.empty:
        print(f"[!] Warning: No data found for Chain '{chain_id}', Residue '{residue_id}'")
        return

    # Sort strictly by frame to ensure correct temporal order
    res_data = res_data.sort_values(by='Frame')
    
    frames = res_data['Frame'].values
    frust_vals = res_data['FrstIndex'].values

    plt.figure(figsize=(20, 4))

    # Line connecting points (the "motion" aspect)
    plt.plot(frames, frust_vals, color='gray', linewidth=0.5, linestyle='-', zorder=1)

    # Reference threshold lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.axhline(y=-1, color='r', linestyle=':', alpha=0.5)
    plt.axhline(y=0.58, color='g', linestyle=':', alpha=0.5)

    # Scatter points with conditional colors based on Frustration Index
    colors = ['green' if y > 0.58 else 'red' if y < -1.0 else 'gray' for y in frust_vals]
    plt.scatter(frames, frust_vals, c=colors, s=20, zorder=2)

    # Styling the plot
    plt.ylim(-4, 2)
    plt.gca().invert_yaxis() # Energy landscapes usually have lower (more stable/frustrated) values down
    
    plt.xlabel('Frame Number', fontweight='bold', fontsize=12)
    plt.ylabel('Frustration Index', fontweight='bold', fontsize=12)
    plt.title(f'Temporal Frustration Dynamics - Residue {residue_id} (Chain {chain_id})', fontsize=16, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Custom legend
    handles = [
        mpatches.Patch(color='green', label='Minimally Frustrated (> 0.58)'),
        mpatches.Patch(color='red', label='Highly Frustrated (< -1.0)'),
        mpatches.Patch(color='gray', label='Neutral')
    ]
    plt.legend(handles=handles, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    
    # Save figure
    out_path = os.path.join(out_dir, f"timeseries_chain_{chain_id}_res_{residue_id}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f" -> Saved plot successfully: {out_path}")
    plt.close()


def main():
    args = parse_arguments()

    print("\n" + "="*50)
    print(" FrustraMotion Visualizer - Temporal Dynamics")
    print("="*50)
    
    # Load the Data
    print(f"[*] Accessing dataframes in: {args.dataframes_dir}")
    df_master = load_tidy_data(args.dataframes_dir, args.chain)
    
    # Determine working chain
    chain_id = args.chain if args.chain else df_master['Chain'].iloc[0]
    
    # Generate the Time-Series Plot
    print(f"[*] Generating Temporal Plot for Residue {args.residue}...")
    plot_frustration_vs_frames(df_master, args.residue, chain_id, args.out_dir)

    print("[*] Process complete!\n")

if __name__ == "__main__":
    main()