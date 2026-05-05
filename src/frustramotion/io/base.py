import pandas as pd
import os
import re
from abc import ABC, abstractmethod
from frustramotion.analysis.single import SingleResidueAnalyzer
from frustramotion.analysis.contact import ContactNetworkAnalyzer

class Base3DExporter(ABC):
    """
    Abstract Base Class for 3D structural export.
    Handles data validation, chain filtering, metric calculation, and normalization.
    """
    def __init__(self, df, pdb_file, output_path, metric='hotspots', chain_id=None):
        self.df = df.copy()
        self.pdb_file = os.path.abspath(pdb_file).replace('\\', '/')
        self.output_path = output_path
        self.metric = metric
        self.is_contacts = 'ResID1' in self.df.columns
        
        # Auto-detect chain if not provided
        if chain_id is None:
            self.chain_id = self.df['Chain'].iloc[0] if not self.is_contacts else self.df['ChainRes1'].iloc[0]
        else:
            self.chain_id = chain_id

        self._validate()

    def _validate(self):
        """Smart validation ensuring data matches the requested metric."""
        if self.metric == 'network' and not self.is_contacts:
            raise ValueError("[!] Error: The 'network' metric requires a Contacts DataFrame, but Single Residue data was provided.")
        
        if self.metric != 'network' and self.is_contacts:
            raise ValueError(f"[!] Error: Metric '{self.metric}' requires a Single Residue DataFrame, but Contact Network data was provided.")

    @staticmethod
    def get_resnum(res_str):
        """Extracts numeric residue ID."""
        match = re.search(r'\d+', str(res_str))
        return match.group() if match else ""

    def get_single_residue_data(self):
        """Calculates and normalizes Single Residue metrics for B-factor injection."""
        chain_data = self.df[self.df['Chain'] == self.chain_id]
        if chain_data.empty:
            raise ValueError(f"[!] Error: No data found for Chain {self.chain_id}")

        traj = SingleResidueAnalyzer(chain_data)
        
        # Map metrics to their calculation functions
        if self.metric == 'hotspots':
            stats = traj.get_hotspots(top_n=9999).reset_index()
            val_col, title = 'Dynamic_Score', "Dynamic Variance (Hotspots)"
        elif self.metric == 'entropy':
            stats = traj.get_state_entropy().reset_index()
            val_col, title = 'Shannon_Entropy', "State Entropy"
        elif self.metric == 'flipping':
            stats = traj.get_flipping_rate().reset_index()
            val_col, title = 'Flipping_Rate', "Flipping Rate"
        else:
            raise ValueError(f"[!] Metric {self.metric} not supported.")

        stats['ResNum'] = stats['Residue'].apply(self.get_resnum)
        
        # Normalize 0-100
        min_val, max_val = stats[val_col].min(), stats[val_col].max()
        if max_val > min_val:
            stats['Bfactor'] = ((stats[val_col] - min_val) / (max_val - min_val)) * 100
        else:
            stats['Bfactor'] = 0
            
        return stats, title

    def get_contact_data(self):
        """Extracts top persistent contact edges for network drawing."""
        net_analyzer = ContactNetworkAnalyzer(self.df)
        stats = net_analyzer.get_contact_persistence(min_frequency=0.2)
        hot_edges = stats[stats['Mean_FrstIndex'] <= -1.0].head(50)
        stable_edges = stats[stats['Mean_FrstIndex'] >= 0.78].head(50)
        return hot_edges, stable_edges

    @abstractmethod
    def export(self):
        """Child classes MUST implement this method."""
        pass
