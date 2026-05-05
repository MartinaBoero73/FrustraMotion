from frustramotion.io.base import Base3DExporter

class ChimeraXExporter(Base3DExporter):
    
    def export(self):
        with open(self.output_path, 'w') as cxc:
            cxc.write(f"# FrustraMotion ChimeraX Export\n")
            cxc.write(f"# Chain: {self.chain_id}\n\n")
            
            # Setup Environment
            cxc.write("set bgColor white\nlighting soft\n\n")
            cxc.write(f"open \"{self.pdb_file}\"\nhide atoms\n")
            cxc.write(f"show /{self.chain_id} cartoon\n\n")

            if self.is_contacts:
                print("[*] Generating ChimeraX Network Pseudobonds...")
                hot_edges, stable_edges = self.get_contact_data()
                
                cxc.write(f"color /{self.chain_id} light gray\n")
                cxc.write(f"transparency /{self.chain_id} 40 target c\n\n")
                
                def draw_edges(edges, color):
                    for pair_id, _ in edges.iterrows():
                        res1_num = self.get_resnum(pair_id.split('_')[0])
                        res2_num = self.get_resnum(pair_id.split('_')[1])
                        if res1_num and res2_num:
                            cxc.write(f"distance /{self.chain_id}:{res1_num}@CA /{self.chain_id}:{res2_num}@CA\n")
                            cxc.write(f"color /{self.chain_id}:{res1_num}@CA /{self.chain_id}:{res2_num}@CA {color} target p\n")

                cxc.write("# --- Highly Frustrated (Red) ---\n")
                draw_edges(hot_edges, "red")
                cxc.write("\n# --- Minimally Frustrated (Green) ---\n")
                draw_edges(stable_edges, "green")
                
                cxc.write("\nhide pbonds label\nsize target p 0.2\n")

            else:
                stats, title = self.get_single_residue_data()
                print(f"[*] Generating ChimeraX B-Factor Heatmap: {title}...")
                
                cxc.write("# Injecting analytical values into B-factor...\n")
                for _, row in stats.iterrows():
                    cxc.write(f"setattr /{self.chain_id}:{row['ResNum']} atoms bfactor {row['Bfactor']:.2f}\n")
                    
                cxc.write(f"\ncolor byattribute bfactor /{self.chain_id} palette blue:white:red\n")

            cxc.write("\nview\n")
        print(f" -> Saved ChimeraX Script: {self.output_path}")
