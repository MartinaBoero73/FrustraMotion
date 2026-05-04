import pandas as pd
import numpy as np

class FrustrationTrajectory:
    """
    Core analytical engine for FrustraMotion.
    Takes a DataFrame (Single Residue or Contacts) and computes advanced biophysical metrics.
    """
    
    def __init__(self, df):
        # Ensure the required base columns exist
        if not all(col in df.columns for col in ['Frame', 'FrstIndex']):
            raise ValueError("[!] Error: DataFrame must contain 'Frame' and 'FrstIndex' columns.")
        
        self.df = df.copy()
        
        # Determine if we are analyzing Single Residue or Contacts
        self.mode = 'contacts' if 'Pair' in df.columns or 'ResID1' in df.columns else 'single'
        self.entity_col = 'Pair' if self.mode == 'contacts' else 'Residue'
        
        # Build the 'Pair' identifier for contacts if it doesn't exist
        if self.mode == 'contacts' and 'Pair' not in self.df.columns:
            self.df['Pair'] = self.df.apply(
                lambda r: f"{r['ResID1']} ↔ {r['ResID2']}" if r['ResID1'] < r['ResID2'] else f"{r['ResID2']} ↔ {r['ResID1']}", 
                axis=1
            )
            
        # Pre-calculate categorical states for advanced metrics
        conditions = [
            (self.df['FrstIndex'] > 0.78),
            (self.df['FrstIndex'] < -1.0)
        ]
        choices = ['Minimally', 'Highly']
        self.df['State'] = np.select(conditions, choices, default='Neutral')

    def get_hotspots(self, top_n=10):
        """
        Calculates the variance (dynamism) of each entity over time.
        Highlights regions with high energetic fluctuations.
        """
        stats = self.df.groupby(self.entity_col)['FrstIndex'].agg(['std', 'mean', 'max', 'min'])
        stats = stats.rename(columns={'std': 'Dynamic_Score'})
        
        # Filter out noise (entities present in less than 10% of the frames)
        counts = self.df.groupby(self.entity_col).size()
        valid_entities = counts[counts > (self.df['Frame'].nunique() * 0.1)].index
        stats = stats.loc[valid_entities]
        
        hotspots = stats.sort_values(by='Dynamic_Score', ascending=False)
        return hotspots.head(top_n)

    def get_dwell_times(self):
        """
        Calculates the % of time each entity spends in each frustration state.
        Sorted by the 'Highly' frustrated percentage in descending order.
        """
        counts = self.df.groupby([self.entity_col, 'State']).size().unstack(fill_value=0)
        
        # Convert to percentages (0-100%)
        dwell_times = counts.div(counts.sum(axis=1), axis=0) * 100
        
        # Ensure all columns exist to prevent KeyError
        for state in ['Minimally', 'Highly', 'Neutral']:
            if state not in dwell_times.columns:
                dwell_times[state] = 0.0
                
        dwell_times = dwell_times[['Minimally', 'Neutral', 'Highly']]
        
        # Sort by Highly frustrated time (descending)
        return dwell_times.sort_values(by='Highly', ascending=False)

    def get_state_entropy(self):
        """
        Calculates the Shannon Entropy of frustration states for each entity.
        High entropy means the residue visits all states equally (highly dynamic switch).
        Low entropy means it is locked in a single energetic state.
        """
        # Get raw probabilities (0 to 1) instead of percentages
        counts = self.df.groupby([self.entity_col, 'State']).size().unstack(fill_value=0)
        probs = counts.div(counts.sum(axis=1), axis=0)
        
        # Calculate Shannon Entropy: H = -sum(p * log2(p))
        # Add a tiny epsilon to avoid log2(0)
        epsilon = 1e-9
        entropy = -(probs * np.log2(probs + epsilon)).sum(axis=1)
        
        entropy_df = pd.DataFrame({'Shannon_Entropy': entropy})
        return entropy_df.sort_values(by='Shannon_Entropy', ascending=False)

    def get_flipping_rate(self):
        """
        Counts how many times an entity transitions from one state to another.
        Normalized by the total number of frames it exists in.
        """
        flip_rates = []
        
        for entity, group in self.df.groupby(self.entity_col):
            group = group.sort_values('Frame')
            states = group['State'].values
            
            if len(states) < 2:
                continue
                
            # Count transitions where current state != previous state
            transitions = np.sum(states[1:] != states[:-1])
            rate = transitions / len(states)
            
            flip_rates.append({
                self.entity_col: entity,
                'Total_Transitions': transitions,
                'Flipping_Rate': rate
            })
            
        return pd.DataFrame(flip_rates).set_index(self.entity_col).sort_values(by='Flipping_Rate', ascending=False)

    def get_persistence(self):
        """
        Finds the maximum continuous streak (in frames) a residue spends 
        trapped in 'Highly' or 'Minimally' frustrated states.
        """
        persistence_data = []
        
        for entity, group in self.df.groupby(self.entity_col):
            group = group.sort_values('Frame')
            states = group['State'].values
            
            # Helper function to find max consecutive identical elements
            def max_consecutive(state_array, target_state):
                mask = (state_array == target_state)
                # Pad with False to easily find boundaries
                padded = np.pad(mask, (1, 1), 'constant', constant_values=False)
                edges = np.diff(padded.astype(int))
                starts = np.where(edges == 1)[0]
                ends = np.where(edges == -1)[0]
                
                if len(starts) == 0:
                    return 0
                return np.max(ends - starts)
            
            max_highly = max_consecutive(states, 'Highly')
            max_minimally = max_consecutive(states, 'Minimally')
            
            persistence_data.append({
                self.entity_col: entity,
                'Max_Streak_Highly': max_highly,
                'Max_Streak_Minimally': max_minimally
            })
            
        return pd.DataFrame(persistence_data).set_index(self.entity_col).sort_values(by='Max_Streak_Highly', ascending=False)

    def detect_transitions(self, window=20, threshold=1.0):
        """
        Scans the trajectory for sudden, sharp energy jumps (conformational shifts).
        Compares the rolling mean of the past window vs the future window.
        """
        transitions = []
        
        for entity, group in self.df.groupby(self.entity_col):
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
                        self.entity_col: entity,
                        'Frame_Trigger': frames[i],
                        'Past_Energy': past_mean,
                        'Future_Energy': future_mean,
                        'Delta': delta
                    })
                    
        if transitions:
            trans_df = pd.DataFrame(transitions)
            return trans_df.sort_values(by='Delta', ascending=False).drop_duplicates(subset=[self.entity_col])
        else:
            return pd.DataFrame()