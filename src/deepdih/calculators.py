from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from ase.calculators.mixing import SumCalculator
from rdkit import Chem
from .settings import settings
from typing import List, Tuple, Dict, Optional
from .utils.geometry import dihedral
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np


EV_TO_KJ_MOL: float = 96.4845044


def merge_calculators(calc1: Calculator, calc2: Calculator) -> Calculator:
    return SumCalculator([calc1, calc2])


class OpenMMBiasCalculator(Calculator):

    name = "OpenMMBias"
    implemented_properties = ["energy", "forces"]

    def __init__(
        self, 
        rdmol: Chem.rdchem.Mol, 
        constraints: List[Tuple[int,int,int,int]] = None, 
        h_bond_repulsion: bool = True, **kwargs
    ):
        Calculator.__init__(self, label=self.name, **kwargs)
        self.rdmol = rdmol

        # create a system
        self.system = mm.System()
        for iatom in range(self.rdmol.GetNumAtoms()):
            atom = self.rdmol.GetAtomWithIdx(iatom)
            mass = atom.GetMass()
            self.system.addParticle(mass)
        
        # create a force
        if h_bond_repulsion:
            h_bond_donors, h_bond_acceptors = [], []
            # find hydrogens linked to N/O/F
            for iatom in range(self.rdmol.GetNumAtoms()):
                atom = self.rdmol.GetAtomWithIdx(iatom)
                if atom.GetAtomicNum() in [7, 8, 9]:
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetAtomicNum() == 1:
                            h_bond_donors.append(iatom)
            # find N/O/F
            for iatom in range(self.rdmol.GetNumAtoms()):
                atom = self.rdmol.GetAtomWithIdx(iatom)
                if atom.GetAtomicNum() in [7, 8, 9]:
                    h_bond_acceptors.append(iatom)
            
            force = mm.CustomBondForce("C6/r^6")
            force.addPerBondParameter("C6")
            for donor in h_bond_donors:
                for acceptor in h_bond_acceptors:
                    # check if the donor and acceptor are bonded
                    if self.rdmol.GetBondBetweenAtoms(donor, acceptor) is None:
                        force.addBond(donor, acceptor, [settings['hbond_repulsion']])
            self.system.addForce(force)

        # create a force
        target_vals = []
        positions = rdmol.GetConformer().GetPositions()
        for ii, jj, kk, ll in constraints:
            dih_val = dihedral(positions[ii], positions[jj], positions[kk], positions[ll])
            target_vals.append((ii, jj, kk, ll, dih_val))
        force = mm.CustomTorsionForce('0.5*k*(theta-theta0)^2')
        force.addPerTorsionParameter('k')
        force.addPerTorsionParameter('theta0')
        
        for ii, jj, kk, ll, target in target_vals:
            force.addTorsion(ii, jj, kk, ll, [settings['relax_torsion_bias'], target / 180 * np.pi])
        self.system.addForce(force)

        # create a integrator
        self.integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
        platform = mm.Platform.getPlatformByName('Reference')
        self.context = mm.Context(self.system, self.integrator, platform)

    def calculate(
        self,
        atoms: Optional["Atoms"] = None,
        properties: List[str] = ["energy", "forces"],
        system_changes: List[str] = all_changes,
    ):
        coord = atoms.get_positions() * 0.1 * unit.nanometer
        self.context.setPositions(coord)
        self.context.computeVirtualSites()
        self.context.applyConstraints(1e-9)
        state = self.context.getState(getForces=True, getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer)

        self.results["energy"] = energy / EV_TO_KJ_MOL
        self.results["forces"] = forces / EV_TO_KJ_MOL * 0.1


class OpenMMGMXTopCalculator(Calculator):

    name = "OpenMMGMXTop"
    implemented_properties = ["energy", "forces"]
    
    def __init__(self, top_file: str, turnoff_propers: bool = False, **kwargs):
        Calculator.__init__(self, label="OpenMMGMXTop", **kwargs)
        self.top_file = top_file
        top = app.GromacsTopFile(self.top_file)
        bond_map = np.zeros((top.getNumBonds(), top.getNumBonds()), dtype=int)
        for bond in top.topology.bonds():
            bond_map[bond[0].index, bond[1].index] = 1
            bond_map[bond[1].index, bond[0].index] = 1
        self.system = top.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)

        if turnoff_propers:
            for force in self.system.getForces():
                if isinstance(force, mm.PeriodicTorsionForce):
                    torsion_force = force
            for ntor in range(torsion_force.getNumTorsions()):
                ii, jj, kk, ll, perodicity, phase, k = torsion_force.getTorsionParameters(ntor)
                if bond_map[ii, jj] > 0 and bond_map[jj, kk] > 0 and bond_map[kk, ll] > 0:
                    torsion_force.setTorsionParameters(ntor, ii, jj, kk, ll, perodicity, phase, 0.0)


        self.integrator = mm.VerletIntegrator(1e-12*unit.femtosecond)
        self.context = mm.Context(self.system, self.integrator, mm.Platform.getPlatformByName('Reference'))