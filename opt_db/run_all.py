# Standard library imports
import os
import warnings
import logging
import subprocess
import platform
from glob import glob
# External library imports
import numpy as np
import tensorflow as tf
# ASE imports
from ase.io import read, write
from ase.db import connect
from ase.atoms import Atoms as Atom
# Pymatgen imports
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
# M3GNet imports
from m3gnet.models import Relaxer, M3GNet, M3GNetCalculator, Potential
# Phonopy imports
from phonopy import Phonopy
from phonopy.phonon import band_structure
# Custom library imports
import ML as ml
# Suppress warnings and TensorFlow GPU usage
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')
tf.config.experimental.set_visible_devices([], 'GPU')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def get_min_frequency(file_path, stru_name):
    min_freq = None
    with open(f"{file_path}/opt_{stru_name}.yaml", "r") as file:
        frequencies = []
        for line in file:
            if "frequency: " in line:
                parts = line.strip().split()
                if parts:
                    freq = parts[-1]
                    try:
                        frequencies.append(float(freq))
                    except ValueError:
                        pass  # Handle the case where conversion to float fails
        if frequencies:
            min_freq = min(frequencies)
    return min_freq

class CIFProcessor:
    def __init__(self, path):
        self.path = path
        self.db_path = f"{self.path}/data.db"
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(filename=rf'{self.path}/map.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    def clean(self):
        if platform.system() == 'Linux' or platform.system() == 'Darwin':
            subprocess.run(f"rm {self.path}/*t*", shell=True, check=True)
        elif platform.system() == 'Windows':
            subprocess.run(f'del "{self.path}\\*t*"', shell=True, check=True)
    def process_files(self):
        logging.info(f"filter is running on {self.path}")
        print(platform.system())
        files = [file for file in os.listdir(self.path) if file.endswith('.cif')]
        new_list = sorted(set([s.split("t")[0] for s in files]))

        logging.info(f"reading {len(new_list)} cif files")
        relaxer = Relaxer()
        for file_name in new_list:
            try:
                energy = {}
                for type in glob(f"{self.path}/{file_name}t*"):
                    structure = Structure.from_file(type)
                    relax_results = relaxer.relax(structure, steps=100, fmax=0.001, verbose=True)
                    final_energy = float(relax_results['trajectory'].energies[-1])
                    force = np.min(relax_results['trajectory'].forces[-1])
                    energy[type] = final_energy/len(structure) + abs(force)

                min_key = min(energy, key=energy.get)
                self.handle_file_operation(file_name, min_key)
                logging.info(f"{file_name} is done")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                logging.error(f"{file_name} is not done due to {e}")

        logging.info("filter is done")
        logging.info("m3gnet opt now !")

    def handle_file_operation(self, file_name, min_key):
        if platform.system() == 'Linux' or platform.system() == 'Darwin':
            subprocess.run(["cp", min_key, f"{self.path}/{file_name}.cif"], check=True)
            subprocess.run(f"rm {self.path}/{file_name}t*", shell=True, check=True)
        elif platform.system() == 'Windows':
            subprocess.run(f'copy "{min_key}" "{self.path}\\{file_name}.cif"', shell=True, check=True)
            subprocess.run(f'del "{self.path}\\{file_name}t*"', shell=True, check=True)




class StructureProcessor:
    def __init__(self, file_path, db_path):
        self.file_path = file_path
        self.db_path = db_path
        self.relaxer = Relaxer()
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(filename=rf'{self.file_path}/processing.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def process_structures(self):
        files = [file for file in os.listdir(self.file_path) if file.endswith('.cif')]
        for file in files:
            stru_name = file.split(".")[0]
            try:
                clean_structure,final_structure,final_energy,force = self.process_structure(stru_name)
                #print(relax_results)
                phonon_data, min_freq, if_w = self.post_process(stru_name, final_structure)
                self.write_to_database_and_file(stru_name, clean_structure, final_energy, force, if_w, min_freq)
                logging.info(f"{stru_name} is done")
            except Exception as e:
                print(f"Error processing {stru_name}: {e}")
                logging.error(f"{stru_name} is not done due to {e}")
                pass

    def process_structure(self, stru_name):
        structure = Structure.from_file(f"{self.file_path}/{stru_name}.cif")
        relax_results = self.relaxer.relax(structure, steps=1000, fmax=0.01, verbose=True)
        for _ in range(50):
            relax_results = self.relaxer.relax(relax_results['final_structure'], steps=300, fmax=0.001, verbose=True)
        relax_results = self.relaxer.relax(relax_results['final_structure'], steps=10000, fmax=0.0001, verbose=True)
        final_structure = self.relaxer.ase_adaptor.get_atoms(relax_results['final_structure'])
        # clear_structure = self.clear_structure(final_structure)
        clean_structure = Atom(cell=final_structure.cell, symbols=final_structure.get_chemical_symbols(), positions=final_structure.get_positions())
        return clean_structure,final_structure,relax_results['trajectory'].energies[-1],np.min(relax_results['trajectory'].forces[-1])

    def post_process(self, stru_name, final_structure):
        phonon_data = self.calculate_phonons(self.relaxer.ase_adaptor.get_structure(final_structure), stru_name)
        min_freq, if_w = self.analyze_phonon_stability(stru_name)
        print(min_freq, if_w)
        return phonon_data, min_freq, if_w

    def calculate_phonons(self, structure, stru_name):
        phonon = ml.get_phonons(structure, supercell_matrix=[2, 2, 2])
        freqs, eigvec = phonon.get_frequencies_with_eigenvectors(q=[0, 0, 0])
        phonon.save(settings={'force_constants': True})
        phonon.auto_band_structure(npoints=40, plot=True, with_eigenvectors=True, write_yaml=True, filename=f"{self.file_path}/opt_{stru_name}.yaml").savefig(f'{self.file_path}/{stru_name}.png', dpi=600)
        return phonon

    def analyze_phonon_stability(self, stru_name):
        if platform.system() == 'Linux' or platform.system() == 'Darwin':
            min_freq = float(os.popen(f'grep "frequency: " {self.file_path}/opt_{stru_name}.yaml | sort -k2 -rn | tail -n1').read().strip().split()[-1])
        elif platform.system() == 'Windows':
            min_freq = get_min_frequency(self.file_path, stru_name)
        if_w = min_freq >= -0.5
        return min_freq, if_w

    def write_to_database_and_file(self, stru_name, clean_structure, final_energy,force,if_w, min_freq):
        write(f"{self.file_path}/opt_{stru_name}.cif",clean_structure)
        with connect(self.db_path) as db:
            if stru_name.isnumeric():
                mpid_value = "CGWGAN_" + str(int(stru_name))
            else:
                mpid_value = "CGWGAN_" + stru_name
            db.write(clean_structure, final_energy=float(final_energy), final_force=float(force),final_phonon=f"{if_w}",mpid = mpid_value,min_freq=float(min_freq))
        with open(f"{self.file_path}/rundata.log", "a") as f:
            f.write(f"{clean_structure.get_chemical_formula()} {self.file_path}/opt_{stru_name}.cif  {min_freq} {if_w}\n")
        if platform.system() == 'Linux' or platform.system() == 'Darwin':
            os.popen(f"rm {self.file_path}/opt_{stru_name}.yaml")
        elif platform.system() == 'Windows':
            subprocess.run(f'del "{self.file_path}\\opt_{stru_name}.yaml"', shell=True, check=True)
        



path = file_path =  "C:/Users/Suth-galaxy/Desktop/ASUGPT/Gen_Crystal/gen_files_9"
db_path = f"{file_path}/data.db"
cif_processor = CIFProcessor(path)
structure_processor = StructureProcessor(file_path, db_path)
cif_processor.process_files()
cif_processor.clean()
structure_processor.process_structures()