#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize contact frustration data from DataFrames.')
    parser.add_argument('dataframes_dir', type=str,
                        help='Directory containing the parsed contact CSV files')
    parser.add_argument('--residue', type=str, required=True,
                        help='Specific residue ID to analyze (Format: Chain:Residue, e.g., A:W45)')
    parser.add_argument('--contact_type', type=str, choices=['all', 'intra', 'inter'], default='all',
                        help='Filter contacts to show only intra-chain or inter-chain (default: all)')
    parser.add_argument('--out_dir', type=str, default='./plots/contacts',
                        help='Directory where the generated plots will be saved (default: ./plots/contacts)')

    args = parser.parse_args()

    if not os.path.isdir(args.dataframes_dir):
        print(f"[!] Error: Directory {args.dataframes_dir} does not exist")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    return args

def load_contact_data(dataframes_dir, residue_id):
    """
    Loads ONLY the files that could contain the target residue
    """
    chain_target = residue_id.split(':')[0]
    compact_id = f"_{chain_target}" if chain_target.islower() else chain_target
    
    print(f"[*] Scanning for contact files involving chain '{chain_target}'...")
    all_dfs = []
    
    for filename in os.listdir(dataframes_dir):
        if filename.endswith('.csv') and '_contacts' in filename:
            # We check if the compact_id is part of the filename (e.g., A_vs_B, or chain_A)
            if f"_{compact_id}_" in filename or f"chain_{compact_id}" in filename:
                filepath = os.path.join(dataframes_dir, filename)
                df = pd.read_csv(filepath)
                all_dfs.append(df)
                
    if not all_dfs:
        print(f"[!] Error: No valid contact files found involving chain '{chain_target}' in {dataframes_dir}")
        sys.exit(1)
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    return master_df

def plot_residue_contacts(contacts, residue_id, out_dir, contact_type):
    """
    Plots a multi-panel dashboard for the contact network of a specific residue
    """
    res_contacts = contacts[(contacts['ResID1'] == residue_id) | (contacts['ResID2'] == residue_id)].copy()
    
    if res_contacts.empty:
        print(f"[!] Warning: No contacts found for residue {residue_id}")
        return

    if contact_type == 'intra':
        res_contacts = res_contacts[res_contacts['ChainRes1'] == res_contacts['ChainRes2']]
    elif contact_type == 'inter':
        res_contacts = res_contacts[res_contacts['ChainRes1'] != res_contacts['ChainRes2']]

    if res_contacts.empty:
        print(f"[!] No {contact_type} contacts found for residue {residue_id}")
        return

    res_contacts['Partner'] = res_contacts.apply(
        lambda r: r['ResID2'] if r['ResID1'] == residue_id else r['ResID1'], axis=1
    )
    
    res_contacts = res_contacts.sort_values(by='Frame')
    frames = np.sort(res_contacts['Frame'].unique())
    
    # Panel 1 Data: Mean Frustration per frame
    mean_frust = res_contacts.groupby('Frame')['FrstIndex'].mean()
    rolling_frust = mean_frust.rolling(window=20, center=True, min_periods=1).mean()

    # Panel 2 Data: Contact counts by type
    # Normalize welltype names just in case
    res_contacts['Welltype'] = res_contacts['Welltype'].str.replace('water-mediated', 'water')
    counts = res_contacts.groupby(['Frame', 'Welltype']).size().unstack(fill_value=0)
    # Ensure all columns exist to avoid plotting errors
    for col in ['short', 'long', 'water']:
        if col not in counts.columns: counts[col] = 0

    # Panel 3 Data: Pivot table for the heatmap (Partner vs Frame)
    pivot = res_contacts.pivot(index='Partner', columns='Frame', values='FrstIndex')

    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [1, 1, 2]}, sharex=True)
    fig.suptitle(f'Microenvironment Dynamics: {residue_id} ({contact_type.upper()} Mode)', fontsize=18, fontweight='bold', y=0.95)

    # --- 1: Network Frustration ---
    ax1.plot(mean_frust.index, mean_frust.values, color='gray', alpha=0.4, label='Mean FrstIndex')
    ax1.plot(rolling_frust.index, rolling_frust.values, color='blue', linewidth=2, label='Rolling Avg (20f)')
    ax1.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax1.invert_yaxis() # Biology standard: negative/stable is down
    ax1.set_ylabel('Avg Frustration\nIndex', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_title("A. Local Network Energetics (Mean Frustration of all contacts)", loc='left', fontweight='bold')

    # --- 2: Contact Composition ---
    # Stacked area plot
    ax2.stackplot(counts.index, counts['short'], counts['long'], counts['water'],
                  labels=['Short', 'Long', 'Water-mediated'],
                  colors=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.8)
    ax2.set_ylabel('Number of\nContacts', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_title("B. Physical Composition of Contacts", loc='left', fontweight='bold')

    # --- 3: Partner Timeline (Heatmap-style Scatter) ---
    partners = pivot.index.tolist()
    
    for i, partner in enumerate(partners):
        partner_data = res_contacts[res_contacts['Partner'] == partner]
        x = partner_data['Frame']
        y = np.full(len(x), i)
        c = partner_data['FrstIndex']
        
        colors = ['green' if val > 0.78 else 'red' if val < -1.0 else 'gray' for val in c]
        
        ax3.scatter(x, y, c=colors, s=30, alpha=0.8, marker='s')

    ax3.set_yticks(range(len(partners)))
    ax3.set_yticklabels(partners, fontsize=9)
    ax3.set_ylabel('Interacting\nPartner', fontweight='bold')
    ax3.set_xlabel('Frame Number', fontweight='bold', fontsize=12)
    ax3.grid(True, linestyle=':', alpha=0.5, axis='y')
    ax3.set_title("C. Interaction Timeline", loc='left', fontweight='bold')

    # Legend for Panel 3
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color='green', label='Minimally Frust.'),
        mpatches.Patch(color='gray', label='Neutral'),
        mpatches.Patch(color='red', label='Highly Frust.')
    ]
    ax3.legend(handles=handles, loc='upper right', title="Interaction State")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # Save
    safe_res = residue_id.replace(':', '_')
    out_path = os.path.join(out_dir, f"contact_dashboard_{safe_res}_{contact_type}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f" -> Saved Contact Dashboard: {out_path}")
    plt.close()

def main():
    args = parse_arguments()

    print("\n" + "="*50)
    print(" FrustraMotion Visualizer - Contact Networks")
    print("="*50)
    
    df_master = load_contact_data(args.dataframes_dir, args.residue)
    plot_residue_contacts(df_master, args.residue, args.out_dir, args.contact_type)

    print("[*] Contact Visualization complete!\n")

if __name__ == "__main__":
    main()