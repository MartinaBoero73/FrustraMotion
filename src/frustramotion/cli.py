import argparse
import sys
import pandas as pd

# Use try/except to catch import errors if the package is not installed correctly
try:
    from frustramotion.parsers.single import parse_frustration_files as parse_single, save_dataframes as save_single
    from frustramotion.parsers.contacts import parse_contact_files as parse_contacts, save_contact_data as save_contacts
    from frustramotion.plotting.timeseries import load_tidy_data, plot_frustration_vs_frames
    from frustramotion.plotting.contacts import load_contact_data, plot_residue_contacts
    from frustramotion.analysis.core import FrustrationTrajectory
    from frustramotion.analysis.contact import ContactNetworkAnalyzer
    from frustramotion.io.export_vmd import generate_vmd_script
    from frustramotion.io.export_chimerax import generate_chimerax_script

except ImportError:
    print("[!] Import Error. Install all the required packages or move to the correct directory.")
    sys.exit(1)

def main():
    """Main entrypoint for the FrustraMotion CLI."""
    
    # Main parser setup
    parser = argparse.ArgumentParser(
        prog='frustramotion',
        description='FrustraMotion: Toolkit for analyzing temporal protein frustration dynamics.'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True

    # ==========================================
    # COMMAND 1: frustramotion parse
    # ==========================================
    parser_parse = subparsers.add_parser('parse', help='Parse raw FrustratometeR output into CSVs')
    parser_parse.add_argument('mode', choices=['single', 'contacts'], help='Type of data to parse')
    parser_parse.add_argument('-i', '--input_dir', required=True, help='Input directory containing .done files')
    parser_parse.add_argument('-o', '--out_dir', default='./parsed_data', help='Output directory for CSVs')
    parser_parse.add_argument('-p', '--protein', default='Protein', help='Prefix name for output files')
    parser_parse.add_argument('-t', '--type', default='configurational', choices=['configurational', 'mutational'], help='(Contacts only) configurational or mutational data')

    # ==========================================
    # COMMAND 2: frustramotion plot
    # ==========================================
    parser_plot = subparsers.add_parser('plot', help='Generate frustration visualizations')
    parser_plot.add_argument('mode', choices=['timeseries', 'contacts'], help='Type of plot to generate')
    parser_plot.add_argument('-i', '--input_dir', required=True, help='Input directory containing FrustraMotion CSVs')
    parser_plot.add_argument('-c', '--chain', help='Specific chain to plot (e.g., A)')
    parser_plot.add_argument('-r', '--residue', required=True, help='Specific residue to plot (e.g., W45 or A:W45 for contacts)')
    parser_plot.add_argument('--rolling', type=int, default=50, help='Rolling average window size (0 to disable)')
    parser_plot.add_argument('-o', '--out_dir', default='./plots', help='Output directory for PNG images')
    parser_plot.add_argument('-t', '--contact_type', choices=['all', 'intra', 'inter'], default='all', help='(Contacts only) Filter contacts to show only intra-chain or inter-chain')

    # ==========================================
    # COMMAND 3: frustramotion analyze
    # ==========================================
    parser_analyze = subparsers.add_parser('analyze', help='Run mathematical and biophysical analysis on trajectories')
    parser_analyze.add_argument('-i', '--input_csv', required=True, help='A specific CSV file (e.g., chain_A.csv)')
    
    # Add ALL metrics (single residue + contacts) to the choices
    parser_analyze.add_argument('metric', 
                                choices=[
                                    # Single Residue Metrics
                                    'hotspots', 'dwell', 'transitions', 'entropy', 'flipping', 'persistence',
                                    # Contact Network Metrics
                                    'hubs', 'hot_edges', 'contact_persistence'
                                ], 
                                help='Which biophysical metric to calculate')
                                
    parser_analyze.add_argument('-o', '--out_csv', default='./analysis_results.csv', help='Where to save the results table')

    # ==========================================
    # COMMAND 4: frustramotion export
    # ==========================================
    parser_export = subparsers.add_parser('export', help='Export 3D analytical heatmaps')
    parser_export.add_argument('software', choices=['vmd', 'chimerax'], help='Target 3D software')
    parser_export.add_argument('-i', '--input_csv', required=True, help='CSV file (e.g., chain_A.csv)')
    parser_export.add_argument('--pdb', required=True, help='Path to the reference PDB file')
    parser_export.add_argument('-m', '--metric', choices=['hotspots', 'entropy', 'flipping'], default='hotspots', help='Which metric to map')
    parser_export.add_argument('-c', '--chain', help='Specific chain to color')
    parser_export.add_argument('-o', '--out', default='./heatmap.cxc', help='Output script path (.cxc for chimerax)')
    # Parse the arguments provided by the user
    args = parser.parse_args()

    # ==========================================
    # EXECUTION LOGIC
    # ==========================================
    
    if args.command == 'parse':
        print(f"\n[*] FrustraMotion Parser - Mode: {args.mode}")
        if args.mode == 'single':
            dfs = parse_single(args.input_dir)
            save_single(dfs, args.out_dir, args.protein)
        elif args.mode == 'contacts':
            df = parse_contacts(args.input_dir, contact_type=args.type)
            save_contacts(df, args.out_dir, args.protein, args.type)
            
    elif args.command == 'plot':
        print(f"\n[*] FrustraMotion Plotter - Mode: {args.mode}")
        
        if args.mode == 'timeseries':
            df = load_tidy_data(args.input_dir, args.chain)
            chain_id = args.chain if args.chain else df['Chain'].iloc[0]
            plot_frustration_vs_frames(df, args.residue, chain_id, args.out_dir, args.rolling)
            
        elif args.mode == 'contacts':
            residue_id = args.residue
            if ':' not in residue_id:
                if args.chain:
                    residue_id = f"{args.chain}:{args.residue}"
                else:
                    print("[!] Error: For contacts plot, specify residue as Chain:ResNum (e.g. -r A:W45) or use the -c flag (e.g. -c A -r W45).")
                    sys.exit(1)
            
            df_master = load_contact_data(args.input_dir, residue_id)
            plot_residue_contacts(df_master, residue_id, args.out_dir, args.contact_type)

    elif args.command == 'analyze':
        print(f"\n[*] FrustraMotion Analytics - Calculating: {args.metric}")
        
        df = pd.read_csv(args.input_csv)
        
        # 1. Auto-detect data type based on columns
        is_contacts_data = 'ResID1' in df.columns
        
        # 2. Route to the appropriate engine
        if not is_contacts_data:
            # --- SINGLE RESIDUE MODE ---
            valid_single_metrics = ['hotspots', 'dwell', 'transitions', 'entropy', 'flipping', 'persistence']
            
            if args.metric not in valid_single_metrics:
                print(f"[!] Error: Metric '{args.metric}' is for Contact Networks, but you provided a Single Residue CSV.")
                sys.exit(1)
                
            print(" -> Detected Single Residue data format.")
            traj = FrustrationTrajectory(df)
            
            if args.metric == 'hotspots':
                result = traj.get_hotspots(top_n=20)
            elif args.metric == 'dwell':
                result = traj.get_dwell_times()
            elif args.metric == 'transitions':
                result = traj.detect_transitions(window=20, threshold=1.0)
            elif args.metric == 'entropy':
                result = traj.get_state_entropy()
            elif args.metric == 'flipping':
                result = traj.get_flipping_rate()
            elif args.metric == 'persistence':
                result = traj.get_persistence()
                
        else:
            # --- CONTACT NETWORKS MODE ---
            valid_contact_metrics = ['hubs', 'hot_edges', 'contact_persistence']
            
            if args.metric not in valid_contact_metrics:
                print(f"[!] Error: Metric '{args.metric}' is for Single Residues, but you provided a Contacts CSV.")
                sys.exit(1)
                
            print(" -> Detected Contact Networks data format.")
            net_analyzer = ContactNetworkAnalyzer(df)
            
            if args.metric == 'hubs':
                result = net_analyzer.get_frustration_hubs(top_n=20)
            elif args.metric == 'hot_edges':
                result = net_analyzer.get_frustration_hot_edges(top_n=20)
            elif args.metric == 'contact_persistence':
                result = net_analyzer.get_contact_persistence()

        # 3. Handle empty results and save
        if result is None or result.empty:
            print("[!] No significant results found for this metric (table is empty).")
        else:
            print("\n=== TOP RESULTS ===")
            print(result.head(10))
            
            result.to_csv(args.out_csv)
            print(f"\n[+] Full results table saved to: {args.out_csv}")

    elif args.command == 'export':
        print(f"\n[*] FrustraMotion Exporter - Software: {args.software}")
        if args.software == 'vmd':
            df = pd.read_csv(args.input_csv)
            generate_vmd_script(df, args.pdb, args.out, metric=args.metric, chain_id=args.chain)
        elif args.software == 'chimerax':
            df = pd.read_csv(args.input_csv)
            out_file = args.out if args.out.endswith('.cxc') else args.out + '.cxc'
            generate_chimerax_script(df, args.pdb, out_file, metric=args.metric, chain_id=args.chain)

if __name__ == '__main__':
    main()