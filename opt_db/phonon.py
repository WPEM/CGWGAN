import warnings
from m3gnet.models import Relaxer
from pymatgen.core import Lattice, Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import Structure
from m3gnet.models import M3GNet, M3GNetCalculator, Potential
from pymatgen.io.cif import CifWriter
import ML as ml
import numpy as np
from phonopy.phonon import band_structure as pnpbs
from pymatgen.core.structure import Structure
from phonopy import Phonopy
import matplotlib
#matplotlib.use('Agg')

#1.relaxtion
##get strc from mp
#mpr = MPRester('0AoezABITGu6l4x9ZK')
#structure = mpr.get_structure_by_material_id(mp_id := "mp-1069538")

##specify location of cif on your device
structure = Structure.from_file("Be12Re.vasp")

#structure_strained =structure.copy()  # We create a copy.
# Create a random strain between -5% and 5% for each direction
#strains = np.random.uniform(low=-0.5, high=0.5, size=3)
#structure_strained.apply_strain(strains)
# In addition to the lattice strains, we also perturb the atoms by a distance of 0.1 angstrom.
#structure_strained.perturb(0.5)

relaxer = Relaxer()  # This loads the default pre-trained model
relax_results =relaxer.relax(structure,steps=10000,fmax=0.0001,verbose=False)

final_structure = relax_results['final_structure']
final_energy = float(relax_results['trajectory'].energies[-1])
print(final_structure)
print(f"Final energy is {final_energy:.9f} eV")
final_structure.to(fmt="POSCAR", filename="Be12Re-opt.vasp")


##2.phonon for relaxed struture
# Specify location of CIF on your device
sturture_phonon= Structure.from_file("Be12Re-opt.vasp")
#print(structure)

# Let's compute the phonon bandstructure using Phonopy and the force fields from M3GNet
size_spcl =2
size_spclz =2
phonon = ml.get_phonons(sturture_phonon, supercell_matrix=[size_spcl,size_spcl,size_spclz])
freqs, eigvec= phonon.get_frequencies_with_eigenvectors(q=[0, 0, 0])

phonon.save(settings={'force_constants': True})

phonon.auto_band_structure()
phonon.auto_band_structure(npoints=40,plot=True,with_eigenvectors=True,write_yaml=True).savefig('Be12Re',dpi=600)