import os
import io
import csv
from Bio import Phylo
import dendropy

# Function to compute RF distance
def rf_distance(tree1_in, tree2_in):
    if not os.path.exists(tree1_in) or not os.path.exists(tree2_in):
        raise FileNotFoundError("One or both tree files do not exist.")

    tree1 = Phylo.read(tree1_in, 'newick')
    tree2 = Phylo.read(tree2_in, 'newick')

    # Normalize clade names in tree2
    for clade in tree2.get_nonterminals():
        if 'inner' in (clade.name or '').lower():
            clade.name = None

    # Prune to common taxa
    taxa1_names = {leaf.name for leaf in tree1.get_terminals()}
    taxa2_names = {leaf.name for leaf in tree2.get_terminals()}
    common_taxa_names = taxa1_names.intersection(taxa2_names)

    for clade in list(tree1.find_clades(terminal=True)):
        if clade.name not in common_taxa_names:
            tree1.prune(clade)

    for clade in list(tree2.find_clades(terminal=True)):
        if clade.name not in common_taxa_names:
            tree2.prune(clade)

    # Convert to DendroPy trees for RF distance calculation
    tree1_newick = io.StringIO()
    tree2_newick = io.StringIO()
    Phylo.write(tree1, tree1_newick, 'newick')
    Phylo.write(tree2, tree2_newick, 'newick')

    taxon_namespace = dendropy.TaxonNamespace()
    tree1_dendropy = dendropy.Tree.get(data=tree1_newick.getvalue(), schema='newick', taxon_namespace=taxon_namespace)
    tree2_dendropy = dendropy.Tree.get(data=tree2_newick.getvalue(), schema='newick', taxon_namespace=taxon_namespace)

    return dendropy.calculate.treecompare.symmetric_difference(tree1_dendropy, tree2_dendropy)

# Paths and protein list
supertree = "/home/rohan/ESM/Data/Trees/OMAM_supertree.nwk"
tree1 = "/home/rohan/ESM/Results/OMAM10_4mammals/Trees/MC1R_4mammals/Consensus_Trees/MC1R_4mammals_astral_consensus_tree.nwk"
tree2 = "/home/rohan/ESM/Results/OMAM10_4mammals/Trees/OPN1SW_4mammals/Consensus_Trees/OPN1SW_4mammals_astral_consensus_tree.nwk"

#print(rf_distance(supertree, tree2))
#print(type(rf_distance(supertree, tree2)))