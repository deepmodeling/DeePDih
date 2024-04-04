from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from ase.calculators.mixing import SumCalculator
from rdkit import Chem
from typing import List, Tuple, Dict, Optional
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import networkx as nx

from ..settings import settings
from ..utils.geometry import dihedral, angle
from ..utils import EV_TO_HARTREE, EV_TO_KJ_MOL, rdmol2graph
from ..utils.topology import getRingAtoms


class OpenMMBiasCalculator(Calculator):

    name = "OpenMMBias"
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        rdmol: Chem.rdchem.Mol,
        restraints: List[Tuple[int, int, int, int]] = [],
        restraint_ring: bool = False,
        h_bond_repulsion: bool = True, **kwargs
    ):
        Calculator.__init__(self, label=self.name, **kwargs)
        self.rdmol = rdmol
        graph = rdmol2graph(rdmol)
        contact_mat = nx.floyd_warshall_numpy(graph)

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
                            h_bond_donors.append(neighbor.GetIdx())
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
                    if contact_mat[donor,acceptor] > 3:
                        force.addBond(donor, acceptor, [
                                      settings['hbond_repulsion']])
            self.system.addForce(force)

        # create a force
        target_vals = []
        positions = rdmol.GetConformer().GetPositions()
        for ii, jj, kk, ll in restraints:
            dih_val = dihedral(
                positions[ii], positions[jj], positions[kk], positions[ll])
            target_vals.append((ii, jj, kk, ll, dih_val))
        
        if len(target_vals) > 0:
            # force = mm.PeriodicTorsionForce()
            force = mm.CustomTorsionForce("0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0); pi = 3.1415926535")
            force.addPerTorsionParameter("theta0")
            force.addPerTorsionParameter("k")
            for ii, jj, kk, ll, target in target_vals:
                force.addTorsion(ii, jj, kk, ll, [target / 180.0 * np.pi, settings['relax_torsion_bias']])
            self.system.addForce(force)

        # restraint 3/4/5/6-membered rings
        if restraint_ring:
            # list ring atoms
            ring_atoms = getRingAtoms(rdmol, join=True)

            # add angle restraint on heavy-heavy-heavy angles
            force = mm.HarmonicAngleForce()
            for atom in rdmol.GetAtoms():
                if atom.GetIdx() not in ring_atoms:
                    continue
                ring_neighbors = []
                not_ring_neighbors = []
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetIdx() in ring_atoms:
                        ring_neighbors.append(neighbor.GetIdx())
                    else:
                        not_ring_neighbors.append(neighbor.GetIdx())
                if len(ring_neighbors) >= 2:
                    for i in range(len(ring_neighbors)):
                        for j in range(i+1, len(ring_neighbors)):
                            angle_val = angle(positions[ring_neighbors[i]], positions[atom.GetIdx()], positions[ring_neighbors[j]])
                            force.addAngle(ring_neighbors[i], atom.GetIdx(), ring_neighbors[j], angle_val, settings['ring_angle_bias'])
                if len(ring_neighbors) >= 1 and len(not_ring_neighbors) >= 1:
                    for i in range(len(ring_neighbors)):
                        for j in range(len(not_ring_neighbors)):
                            angle_val = angle(positions[ring_neighbors[i]], positions[atom.GetIdx()], positions[not_ring_neighbors[j]])
                            force.addAngle(ring_neighbors[i], atom.GetIdx(), not_ring_neighbors[j], angle_val, settings['ring_angle_bias'])
            self.system.addForce(force)

            # add ring atom restraints
            # 1. restraint dihedrals in which the ring bonds are rotamers
            ring_rotamers = []
            for bond in rdmol.GetBonds():
                i1, i2 = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
                if i1 in ring_atoms and i2 in ring_atoms:
                    ring_rotamers.append((i1, i2))
            
            ring_dihedrals = []
            for i1, i2 in ring_rotamers:
                a1 = self.rdmol.GetAtomWithIdx(i1)
                a2 = self.rdmol.GetAtomWithIdx(i2)
                for n1 in a1.GetNeighbors():
                    if n1.GetIdx() != i2:
                        for n2 in a2.GetNeighbors():
                            if n2.GetIdx() != i1:
                                if n1.GetAtomicNum() == 1 and n2.GetAtomicNum() == 1:
                                    continue
                                dih_val = dihedral(
                                    positions[n1.GetIdx()], positions[i1], positions[i2], positions[n2.GetIdx()])
                                ring_dihedrals.append((n1.GetIdx(), i1, i2, n2.GetIdx(), dih_val))

            force = mm.CustomTorsionForce("0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0); pi = 3.1415926535")
            force.addPerTorsionParameter("theta0")
            force.addPerTorsionParameter("k")
            for ii, jj, kk, ll, target in ring_dihedrals:
                force.addTorsion(ii, jj, kk, ll, [target / 180.0 * np.pi, settings['ring_torsion_bias']])
            self.system.addForce(force)

            # force = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            # force.addPerParticleParameter("k")
            # force.addPerParticleParameter("x0")
            # force.addPerParticleParameter("y0")
            # force.addPerParticleParameter("z0")
            # for iatom in ring_atoms:
            #     atom = self.rdmol.GetAtomWithIdx(iatom)
            #     if atom.GetAtomicNum() > 1:
            #         force.addParticle(
            #             iatom, 
            #             [
            #                 settings['ring_atom_rmsd'], 
            #                 positions[iatom][0]*0.1, 
            #                 positions[iatom][1]*0.1, 
            #                 positions[iatom][2]*0.1
            #             ]
            #         )
            # self.system.addForce(force)



        # create a integrator
        self.integrator = mm.VerletIntegrator(1e-12*unit.picoseconds)
        platform = mm.Platform.getPlatformByName('Reference')
        self.context = mm.Context(self.system, self.integrator, platform)

    def calculate(
        self,
        atoms: Optional["Atoms"] = None,
        properties: List[str] = ["energy", "forces"],
        system_changes: List[str] = all_changes,
    ):
        coord = atoms.get_positions() * unit.angstrom
        self.context.setPositions(coord)
        self.context.computeVirtualSites()
        self.context.applyConstraints(1e-9)
        state = self.context.getState(getForces=True, getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer)

        self.results["energy"] = energy / EV_TO_KJ_MOL
        self.results["forces"] = forces / EV_TO_KJ_MOL * 0.1
