import pandas as pd
import numpy as np
from frustramotion.analysis.base import BaseAnalyzer

class SingleResidueAnalyzer(BaseAnalyzer):
    """
    Analytical engine for FrustraMotion (Single Residue Level).
    Computes advanced biophysical metrics for individual amino acids.
    """
    
    def _validate_columns(self):
        # 1. Run the base validation first (checks for Frame and FrstIndex)
        super()._validate_columns()
        
        # 2. Add specific validation for single residues
        if 'Residue' not in self.df.columns:
            raise ValueError("[!] Error: Single residue data must contain a 'Residue' column.")
        
        # 3. Pre-calculate categorical states for advanced metrics
        conditions = [
            (self.df['FrstIndex'] > 0.78),
            (self.df['FrstIndex'] < -1.0)
        ]
        self.df['State'] = np.select(conditions, ['Minimally', 'Highly'], default='Neutral')

    def get_hotspots(self, top_n=10):
        stats = self.df.groupby('Residue')['FrstIndex'].agg(['std', 'mean', 'max', 'min'])
        stats = stats.rename(columns={'std': 'Dynamic_Score'})
        
        # We use the inherited helper method here!
        counts = self.df.groupby('Residue').size()
        valid_residues = counts[counts > self._get_noise_threshold(0.1)].index
        stats = stats.loc[valid_residues]
        
        hotspots = stats.sort_values(by='Dynamic_Score', ascending=False)
        return hotspots.head(top_n)

    def get_dwell_times(self):
        counts = self.df.groupby(['Residue', 'State']).size().unstack(fill_value=0)
        dwell_times = counts.div(counts.sum(axis=1), axis=0) * 100
        
        for state in ['Minimally', 'Highly', 'Neutral']:
            if state not in dwell_times.columns:
                dwell_times[state] = 0.0
                
        dwell_times = dwell_times[['Minimally', 'Neutral', 'Highly']]
        return dwell_times.sort_values(by='Highly', ascending=False)

    def get_state_entropy(self):
        counts = self.df.groupby(['Residue', 'State']).size().unstack(fill_value=0)
        probs = counts.div(counts.sum(axis=1), axis=0)
        
        epsilon = 1e-9
        entropy = -(probs * np.log2(probs + epsilon)).sum(axis=1)
        
        entropy_df = pd.DataFrame({'Shannon_Entropy': entropy})
        return entropy_df.sort_values(by='Shannon_Entropy', ascending=False)

    def get_flipping_rate(self):
        flip_rates = []
        for residue, group in self.df.groupby('Residue'):
            group = group.sort_values('Frame')
            states = group['State'].values
            
            if len(states) < 2:
                continue
                
            transitions = np.sum(states[1:] != states[:-1])
            rate = transitions / len(states)
            
            flip_rates.append({
                'Residue': residue,
                'Total_Transitions': transitions,
                'Flipping_Rate': rate
            })
            
        return pd.DataFrame(flip_rates).set_index('Residue').sort_values(by='Flipping_Rate', ascending=False)

    def get_persistence(self):
        persistence_data = []
        for residue, group in self.df.groupby('Residue'):
            group = group.sort_values('Frame')
            states = group['State'].values
            
            def max_consecutive(state_array, target_state):
                mask = (state_array == target_state)
                padded = np.pad(mask, (1, 1), 'constant', constant_values=False)
                edges = np.diff(padded.astype(int))
                starts = np.where(edges == 1)[0]
                ends = np.where(edges == -1)[0]
                if len(starts) == 0: return 0
                return np.max(ends - starts)
            
            max_highly = max_consecutive(states, 'Highly')
            max_minimally = max_consecutive(states, 'Minimally')
            
            persistence_data.append({
                'Residue': residue,
                'Max_Streak_Highly': max_highly,
                'Max_Streak_Minimally': max_minimally
            })
            
        return pd.DataFrame(persistence_data).set_index('Residue').sort_values(by='Max_Streak_Highly', ascending=False)

    def detect_transitions(self, window=20, threshold=1.0):
        transitions = []
        for residue, group in self.df.groupby('Residue'):
            group = group.sort_values('Frame')
            frst = group['FrstIndex'].values
            frames = group['Frame'].values
            
            if len(frst) < window * 2:
                continue
                
            for i in range(window, len(frst) - window):
                past_mean = np.mean(frst[i-window : i])
                future_mean = np.mean(frst[i : i+window])
                delta = abs(future_mean - past_mean)
                
                if delta >= threshold:
                    transitions.append({
                        'Residue': residue,
                        'Frame_Trigger': frames[i],
                        'Past_Energy': past_mean,
                        'Future_Energy': future_mean,
                        'Delta': delta
                    })
                    
        if transitions:
            trans_df = pd.DataFrame(transitions)
            return trans_df.sort_values(by='Delta', ascending=False).drop_duplicates(subset=['Residue'])
        return pd.DataFrame()