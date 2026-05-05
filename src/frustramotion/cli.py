import argparse
import sys
import pandas as pd

# Use try/except to catch import errors if the package is not installed correctly
try:
    from frustramotion.parsers.single import parse_frustration_files as parse_single, save_dataframes as save_single
    from frustramotion.parsers.contacts import parse_contact_files as parse_contacts, save_contact_data as save_contacts
    from frustramotion.plotting.timeseries import load_tidy_data, plot_frustration_vs_frames
    from frustramotion.plotting.contacts import load_contact_data, plot_residue_contacts
    from frustramotion.analysis.single import SingleResidueAnalyzer
    from frustramotion.analysis.contact import ContactNetworkAnalyzer
    from frustramotion.io.export_vmd import generate_vmd_script
    from frustramotion.io.export_chimerax import generate_chimerax_script

except ImportError:
    print("[!] Import Error. Install all the required packages or move to the correct directory.")
    sys.exit(1)

# ==========================================
# DISPATCH DICTIONARIES (factory)
# ==========================================
SINGLE_METRICS = {
    'hotspots': 'get_hotspots',
    'dwell': 'get_dwell_times',
    'transitions': 'detect_transitions',
    'entropy': 'get_state_entropy',
    'flipping': 'get_flipping_rate',
    'persistence': 'get_persistence'
}

CONTACT_METRICS = {
    'hubs': 'get_frustration_hubs',
    'hot_edges': 'get_frustration_hot_edges',
    'contact_persistence': 'get_contact_persistence'
}


# ==========================================
# COMMAND HANDLERS (Lógica delegada)
# ==========================================
def handle_parse(args):
    print(f"\n[*] FrustraMotion Parser - Mode: {args.mode}")
    if args.mode == 'single':
        dfs = parse_single(args.input_dir)
        save_single(dfs, args.out_dir, args.protein)
    elif args.mode == 'contacts':
        df = parse_contacts(args.input_dir, contact_type=args.type)
        save_contacts(df, args.out_dir, args.protein, args.type)

def handle_plot(args):
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
                print("[!] Error: For contacts plot, specify residue as Chain:ResNum (e.g. -r A:W45) or use the -c flag.")
                sys.exit(1)
        
        df_master = load_contact_data(args.input_dir, residue_id)
        plot_residue_contacts(df_master, residue_id, args.out_dir, args.contact_type)

def handle_analyze(args):
    print(f"\n[*] FrustraMotion Analytics - Calculating: {args.metric}")
    df = pd.read_csv(args.input_csv)
    is_contacts_data = 'ResID1' in df.columns

    # 1. Routing y validación inteligente
    if not is_contacts_data:
        if args.metric not in SINGLE_METRICS:
            print(f"[!] Error: Metric '{args.metric}' is for Contact Networks, but you provided a Single Residue CSV.")
            sys.exit(1)
            
        print(" -> Detected Single Residue data format.")
        analyzer = SingleResidueAnalyzer(df)
        method_to_call = getattr(analyzer, SINGLE_METRICS[args.metric])
        
    else:
        if args.metric not in CONTACT_METRICS:
            print(f"[!] Error: Metric '{args.metric}' is for Single Residues, but you provided a Contacts CSV.")
            sys.exit(1)
            
        print(" -> Detected Contact Networks data format.")
        analyzer = ContactNetworkAnalyzer(df)
        method_to_call = getattr(analyzer, CONTACT_METRICS[args.metric])

    # 2. Ejecución dinámica (Factory)
    # Pasamos parámetros por defecto si la función los requiere
    if args.metric == 'transitions':
        result = method_to_call(window=20, threshold=1.0)
    elif args.metric in ['hotspots', 'hubs', 'hot_edges']:
        result = method_to_call(top_n=20)
    else:
        result = method_to_call()

    # 3. Manejo de resultados
    if result is None or result.empty:
        print("[!] No significant results found for this metric (table is empty).")
    else:
        print("\n=== TOP RESULTS ===")
        print(result.head(10))
        result.to_csv(args.out_csv)
        print(f"\n[+] Full results table saved to: {args.out_csv}")

def handle_export(args):
    print(f"\n[*] FrustraMotion Exporter - Software: {args.software}")
    df = pd.read_csv(args.input_csv)
    if args.software == 'vmd':
        generate_vmd_script(df, args.pdb, args.out, metric=args.metric, chain_id=args.chain)
    elif args.software == 'chimerax':
        out_file = args.out if args.out.endswith('.cxc') else args.out + '.cxc'
        generate_chimerax_script(df, args.pdb, out_file, metric=args.metric, chain_id=args.chain)


# ==========================================
# MAIN CLI ROUTER
# ==========================================
def main():
    """Main entrypoint for the FrustraMotion CLI."""
    
    parser = argparse.ArgumentParser(
        prog='frustramotion',
        description='FrustraMotion: Toolkit for analyzing temporal protein frustration dynamics.'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True

    # --- PARSE ---
    parser_parse = subparsers.add_parser('parse', help='Parse raw FrustratometeR output into CSVs')
    parser_parse.add_argument('mode', choices=['single', 'contacts'], help='Type of data to parse')
    parser_parse.add_argument('-i', '--input_dir', required=True, help='Input directory containing .done files')
    parser_parse.add_argument('-o', '--out_dir', default='./parsed_data', help='Output directory for CSVs')
    parser_parse.add_argument('-p', '--protein', default='Protein', help='Prefix name for output files')
    parser_parse.add_argument('-t', '--type', default='configurational', choices=['configurational', 'mutational'], help='(Contacts) config/mut data')

    # --- PLOT ---
    parser_plot = subparsers.add_parser('plot', help='Generate frustration visualizations')
    parser_plot.add_argument('mode', choices=['timeseries', 'contacts'], help='Type of plot to generate')
    parser_plot.add_argument('-i', '--input_dir', required=True, help='Input directory containing FrustraMotion CSVs')
    parser_plot.add_argument('-c', '--chain', help='Specific chain to plot (e.g., A)')
    parser_plot.add_argument('-r', '--residue', required=True, help='Specific residue to plot (e.g., W45 or A:W45)')
    parser_plot.add_argument('--rolling', type=int, default=50, help='Rolling average window size (0 to disable)')
    parser_plot.add_argument('-o', '--out_dir', default='./plots', help='Output directory for PNG images')
    parser_plot.add_argument('-t', '--contact_type', choices=['all', 'intra', 'inter'], default='all', help='(Contacts) intra/inter filter')

    # --- ANALYZE ---
    parser_analyze = subparsers.add_parser('analyze', help='Run mathematical and biophysical analysis on trajectories')
    parser_analyze.add_argument('-i', '--input_csv', required=True, help='A specific CSV file (e.g., chain_A.csv)')
    parser_analyze.add_argument('metric', 
                                choices=list(SINGLE_METRICS.keys()) + list(CONTACT_METRICS.keys()), 
                                help='Which biophysical metric to calculate')
    parser_analyze.add_argument('-o', '--out_csv', default='./analysis_results.csv', help='Where to save the results table')

    # --- EXPORT ---
    parser_export = subparsers.add_parser('export', help='Export 3D analytical heatmaps')
    parser_export.add_argument('software', choices=['vmd', 'chimerax'], help='Target 3D software')
    parser_export.add_argument('-i', '--input_csv', required=True, help='CSV file (e.g., chain_A.csv)')
    parser_export.add_argument('--pdb', required=True, help='Path to the reference PDB file')
    parser_export.add_argument('-m', '--metric', choices=['hotspots', 'entropy', 'flipping'], default='hotspots', help='Which metric to map')
    parser_export.add_argument('-c', '--chain', help='Specific chain to color')
    parser_export.add_argument('-o', '--out', default='./heatmap.cxc', help='Output script path')

    args = parser.parse_args()

    # Route to the appropriate handler
    if args.command == 'parse':
        handle_parse(args)
    elif args.command == 'plot':
        handle_plot(args)
    elif args.command == 'analyze':
        handle_analyze(args)
    elif args.command == 'export':
        handle_export(args)

if __name__ == '__main__':
    main()