import numpy as np
import loggin
from typing import Callable, Union, List
from monty.json import MSONable, MontyDecoder
from ase.calculators.calculator import Calculator
from ase.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure
from pymatgen.io.ase import AseAtomsAdaptor
from m3gnet.models import M3GNet, Potential, Relaxer, M3GNetCalculator
from phonopy import Phonopy

logger = logging.getLogger(__name__)

class Factory(MSONable):

    def __init__(self, callable: Callable, set_atoms=False, *args, **kwargs):
        self.callable = callable
        self.module = callable.__module__
        self.name = callable.__name__
        self.set_atoms = set_atoms
        self.args = args
        self.kwargs = kwargs

    def generate(self, atoms=None, **kwargs):
        total_kwargs = dict(self.kwargs)
        total_kwargs.update(kwargs)
        if self.set_atoms:
            if not atoms:
                raise RuntimeError("atoms argument is required")
            if "atoms" not in total_kwargs:
                total_kwargs["atoms"] = atoms

        return self.callable(*self.args, **total_kwargs)

    def as_dict(self) -> dict:
        def recursive_as_dict(obj):
            if isinstance(obj, (list, tuple)):
                return [recursive_as_dict(it) for it in obj]
            if isinstance(obj, dict):
                return {kk: recursive_as_dict(vv) for kk, vv in obj.items()}
            if hasattr(obj, "as_dict"):
                return obj.as_dict()
            return obj

        d = {
            "set_atoms": self.set_atoms,
            "module": self.module,
            "name": self.name,
            "args": recursive_as_dict(self.args),
            "kwargs": recursive_as_dict(self.kwargs),
        }

        return d

    @classmethod
    def from_dict(cls, d: dict):
        module = d["module"]
        name = d["name"]
        mod = __import__(module, globals(), locals(), [name], 0)
        callable = getattr(mod, name)
        arg_decoded = MontyDecoder().process_decoded(d["args"])
        kwarg_decoded = MontyDecoder().process_decoded(d["kwargs"])
        set_atoms = d["set_atoms"]

        return cls(callable=callable, set_atoms=set_atoms, *arg_decoded, **kwarg_decoded)

def get_energy_per_atom(structure):
    atoms = AseAtomsAdaptor().get_atoms(structure)
    atoms.calc = M3GNetCalculator(potential = Potential(M3GNet.load()))
    return atoms.get_potential_energy()/len(structure)

def get_relaxed_structure(structure):
    relaxer = Relaxer()
    relax_results = relaxer.relax(structure, verbose=False)
    return relax_results['final_structure']

#def get_phonons(structure: Structure, calculator: Union[Calculator, Factory], constraints: list = None,
def get_phonons(structure: Structure,
#               calculator: Union[Calculator, Factory],
                constraints: list = None,
                supercell_matrix: List[List[int]] = None,
                primitive_matrix: List[List[float]] = None) -> Phonopy:
    unitcell = get_phonopy_structure(structure)
    if supercell_matrix is None:
        supercell_matrix = np.eye(3)

    calculator = M3GNetCalculator(potential=Potential(M3GNet.load()))
    if isinstance(calculator, Factory):
        calculator = calculator.generate()

    phonon = Phonopy(unitcell,
                     supercell_matrix=supercell_matrix,
                     primitive_matrix=primitive_matrix)
    phonon.generate_displacements(distance=0.03)
    supercells = phonon.supercells_with_displacements
    supercells_atoms = []
    for sc in supercells:
        a = Atoms(symbols=sc.symbols,
                  positions=sc.positions,
                  masses=sc.masses,
                  cell=sc.cell, pbc=True,
                  constraint=None,
                  calculator=calculator)
        if constraints:
            tmp_constraints = []
            for i, c in enumerate(constraints):
                if isinstance(c, Factory):
                    tmp_constraints.append(c.generate(atoms=a))
                else:
                    tmp_constraints.append(c)
            a.set_constraint(tmp_constraints)
        supercells_atoms.append(a)

    forces = []
    for i, sca in enumerate(supercells_atoms):
        logger.debug(f"calculating forces for supercell {i+1} of {len(supercells_atoms)}")
        forces.append(sca.get_forces())

    phonon.set_forces(forces)
    phonon.produce_force_constants()
    return phonon
