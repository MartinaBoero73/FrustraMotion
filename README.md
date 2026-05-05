# FrustraMotion

**FrustraMotion** is a Python-based toolkit designed to parse, analyze, and visualize the temporal dynamics of protein frustration networks from Molecular Dynamics (MD) simulations. 

It acts as a downstream processing pipeline for the [FrustratometeR](http://frustratometer.qb.fcen.uba.ar/), transforming hundreds of raw, static frames into continuous time-series data. Whether you are hunting for allosteric switches, tracking solvation dynamics, or mapping energetic variance onto 3D structures, FrustraMotion provides a clean, modular API and a powerful Command Line Interface (CLI) to accelerate your biophysical research.

---

## Key Features

*   **Tidy Data Parsing:** Converts raw `.done` files from the FrustratometeR into optimized, chain-separated Pandas DataFrames ready for analysis.
*   **Single-Residue Analytics:** Detect "Hotspots" by calculating energetic variance (Dynamic Score), Shannon entropy, state dwell times, and transition rates.
*   **Contact Network Dynamics:** Go beyond simple averages with the `ContactNetworkAnalyzer`. Identify structural stress centers ("Frustration Hubs"), track persistently frustrated interactions ("Hot Edges"), and monitor solvation flipping (water-mediated vs. direct contacts).
*   **Publication-Ready Dashboards:** Generate multi-panel visualizations of single-residue energy landscapes and local contact microenvironments directly from the CLI.
*   **VMD and ChimeraX Integration:** Export thermodynamic state metrics directly into the B-factor column of a PDB file, automatically generating `.tcl` or `.cxc` scripts for 3D structural mapping.

---

## Installation

FrustraMotion requires Python 3.8 or higher. 

Clone the repository and install it locally using `pip`:

```bash
git clone https://github.com/MartinaBoero73/FrustraMotion.git
cd FrustraMotion
pip install -e .
```
---


## Quickstart (CLI)

FrustraMotion comes with a built-in CLI for rapid analysis.

1. Parse raw FrustratometeR data into CSVs:

```
frustramotion parse -i ./raw_data -o ./parsed_data -p MyProtein
```

2. Plot Single Residue Dynamics:

```
frustramotion plot timeseries -i ./parsed_data -c A -r W45 -o ./plots
```

3. Plot a Contact Dashboard:

```
frustramotion plot contacts -i ./parsed_data -r A:W45 -t intra -o ./plots
```

4. Get metrics:

```
frustramotion analyze hotspots -i ./parsed_data -o ./results_hotspots.csv hotspots
```

5. Export Data to VMD or ChimeraX:

```
frustramotion export vmd -i ./parsed_data -p ./reference.pdb -m entropy -c A -o ./outputs
```

## User Guide & Tutorial

For a deep dive into the Python API and biophysical analysis, check out the interactive Jupyter Notebook guide included in the repository. It walks through a complete use case analyzing the encapsulin of Mycobacterium tuberculosis (MtEnc).

Open the Interactive [Guide](guides/user_guide.ipynb)


## Project Structure

```
FrustraMotion/
├── src/
│   └── frustramotion/
│       ├── analysis/    # Metric engines 
│       ├── parsers/     # Data wrangling from raw to CSVs
│       ├── plotting/    # Matplotlib/Seaborn dashboard generators
│       ├── io/          # 3D visualization exporters
│       └── cli.py       # Command Line Interface logic
├── guides/              # Jupyter Notebook tutorials and example data
├── setup.py             # Package installation config
└── README.md            
```

## License

This project is open-source and available under the MIT License.