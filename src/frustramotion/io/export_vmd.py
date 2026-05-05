from frustramotion.io.base import Base3DExporter

class VMDExporter(Base3DExporter):
    
    def export(self):
        with open(self.output_path, 'w') as tcl:
            tcl.write(f"# FrustraMotion VMD Export\n")
            tcl.write(f"# Chain: {self.chain_id}\n\n")
            
            # Setup Environment
            tcl.write(f"mol new {{{self.pdb_file}}} type pdb waitfor all\n")
            tcl.write("color Display Background white\ndisplay projection Orthographic\naxes location off\n\n")

            if self.is_contacts:
                print("[*] Generating VMD Network Lines...")
                hot_edges, stable_edges = self.get_contact_data()

                tcl.write("mol delrep 0 top\n")
                tcl.write("mol representation NewCartoon 0.15 10.0 4.1\n")
                tcl.write("mol color colorID 8\n")
                tcl.write(f"mol selection \"chain {self.chain_id}\"\n")
                tcl.write("mol material Opaque\nmol addrep top\n\ndraw delete all\n\n")

                counter = 0
                def draw_edges(edges, color):
                    nonlocal counter
                    for pair_id, _ in edges.iterrows():
                        res1_num = self.get_resnum(pair_id.split('_')[0])
                        res2_num = self.get_resnum(pair_id.split('_')[1])
                        if res1_num and res2_num:
                            tcl.write(f'set sel{counter}a [atomselect top "resid {res1_num} and name CA and chain {self.chain_id}"]\n')
                            tcl.write(f'set sel{counter}b [atomselect top "resid {res2_num} and name CA and chain {self.chain_id}"]\n')
                            tcl.write(f'lassign [$sel{counter}a get {{x y z}}] pos1\nlassign [$sel{counter}b get {{x y z}}] pos2\n')
                            tcl.write(f'draw color {color}\ndraw line $pos1 $pos2 style solid width 2\n')
                            tcl.write(f'$sel{counter}a delete\n$sel{counter}b delete\n\n')
                            counter += 1

                tcl.write("# --- Highly Frustrated (Red) ---\n")
                draw_edges(hot_edges, "red")
                tcl.write("# --- Minimally Frustrated (Green) ---\n")
                draw_edges(stable_edges, "green")

            else:
                stats, title = self.get_single_residue_data()
                print(f"[*] Generating VMD B-Factor Heatmap: {title}...")
                
                tcl.write("set all_atoms [atomselect top all]\n$all_atoms set beta 0\n\n")
                for _, row in stats.iterrows():
                    tcl.write(f"set sel [atomselect top \"chain {self.chain_id} and resid {row['ResNum']}\"]\n")
                    tcl.write(f"$sel set beta {row['Bfactor']:.2f}\n$sel delete\n")
                
                tcl.write("$all_atoms delete\n\n")
                tcl.write("mol delrep 0 top\nmol representation NewCartoon 0.30 10.0 4.1\n")
                tcl.write(f"mol color Beta\nmol selection \"chain {self.chain_id}\"\nmol material Opaque\nmol addrep top\n\n")
                tcl.write("color scale method BWR\ncolor scale midpoint 0.5\n")

            tcl.write("display resetview\n")
        print(f" -> Saved VMD Script: {self.output_path}")