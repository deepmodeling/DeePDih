from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from rdkit import Chem
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
from typing import List, Tuple, Dict, Optional
from ..utils import EV_TO_HARTREE, EV_TO_KJ_MOL


class GromacsTopCalculator(Calculator):

    name = "GromacsTop"
    implemented_properties = ["energy", "forces"]
    
    def __init__(self, rdmol: Chem.rdchem.Mol, top_file: str, turnoff_propers: List[Tuple[int,int,int,int]] = [], **kwargs):
        Calculator.__init__(self, label="GromacsTop", **kwargs)
        self.top_file = top_file
        top = app.GromacsTopFile(self.top_file)
        bond_map = np.zeros((top.topology.getNumAtoms(), top.topology.getNumAtoms()), dtype=int)
        for bond in top.topology.bonds():
            bond_map[bond[0].index, bond[1].index] = 1
            bond_map[bond[1].index, bond[0].index] = 1
        self.system = top.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)

        num_real = rdmol.GetNumAtoms()
        num_particles = self.system.getNumParticles()
        if num_real == num_particles:
            self.real2whole = {iatom: iatom for iatom in range(num_real)}
            self.whole2real = {iatom: iatom for iatom in range(num_real)}
        else:
            self.real2whole = {}
            self.whole2real = {}
            ireal = 0
            for iatom in range(num_particles):
                mass = self.system.getParticleMass(iatom)
                if mass.value_in_unit(unit.amu) > 0:
                    self.real2whole[ireal] = iatom
                    self.whole2real[iatom] = ireal
                    ireal += 1

        for force in self.system.getForces():
            if isinstance(force, mm.PeriodicTorsionForce):
                torsion_force = force
        for proper in turnoff_propers:
            prop_key = (self.real2whole[proper[0]], self.real2whole[proper[1]], self.real2whole[proper[2]], self.real2whole[proper[3]])
            for iterm in range(torsion_force.getNumTorsions()):
                torsion = torsion_force.getTorsionParameters(iterm)
                if (torsion[0], torsion[1], torsion[2], torsion[3]) == prop_key or (torsion[3], torsion[2], torsion[1], torsion[0]) == prop_key:
                    torsion_force.setTorsionParameters(iterm, torsion[0], torsion[1], torsion[2], torsion[3], torsion[4], 0.0, 0.0)

        self.integrator = mm.VerletIntegrator(1e-12*unit.femtosecond)
        self.context = mm.Context(self.system, self.integrator, mm.Platform.getPlatformByName('CPU'))


    def calculate(
        self,
        atoms: Optional["Atoms"] = None,
        properties: List[str] = ["energy", "forces"],
        system_changes: List[str] = all_changes,
    ):
        coord = atoms.get_positions() * 0.1
        coord_all = np.zeros((self.system.getNumParticles(), 3))
        for iatom in range(atoms.get_number_of_atoms()):
            coord_all[self.real2whole[iatom]] = coord[iatom]
        self.context.setPositions(coord_all)
        self.context.computeVirtualSites()
        self.context.applyConstraints(1e-9)
        state = self.context.getState(getForces=True, getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.angstrom)
        force_real = np.zeros((atoms.get_number_of_atoms(), 3))
        for iatom in range(atoms.get_number_of_atoms()):
            force_real[iatom] = forces[self.real2whole[iatom]]

        self.results["energy"] = energy / EV_TO_KJ_MOL
        self.results["forces"] = forces / EV_TO_KJ_MOL