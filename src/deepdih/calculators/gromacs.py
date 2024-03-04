from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np


class GromacsTopCalculator(Calculator):

    name = "GromacsTop"
    implemented_properties = ["energy", "forces"]
    
    def __init__(self, top_file: str, turnoff_propers: bool = False, **kwargs):
        Calculator.__init__(self, label="GromacsTop", **kwargs)
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