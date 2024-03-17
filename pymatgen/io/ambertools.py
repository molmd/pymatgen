# coding: utf-8
from __future__ import division, print_function, unicode_literals, absolute_import

"""
This module is a wrapper for AntechamberRunner which generates force field files
or a specified molecule using gaussian output file as input. Currently, the AntechamberRunner
class does not work properly.
"""

import shlex
import subprocess
import tempfile
import numpy as np
import parmed as pmd
from shutil import which
from collections import namedtuple, OrderedDict

from monty.dev import requires
from monty.tempfile import ScratchDir

from pymatgen.core.structure import Molecule
from pymatgen.io.lammps.data import Topology, ForceField

__author__ = 'Navnidhi Rajput, Kiran Mathew, Matthew Bliss'
__copyright__ = 'Copyright 2013, The Materials Virtual Lab'
__version__ = '0.1'
__maintainer__ = 'Matthew Bliss'
__email__ = 'mbliss01@tufts.edu'
__date__ = '1/29/20'


GAFF_DICT = {'c': '12.01', 'c1': '12.01', 'c2': '12.01', 'c3': '12.01', 'ca': '12.01', 'cp': '12.01', 'cq': '12.01',
             'cc': '12.01', 'cd': '12.01', 'ce': '12.01', 'cf': '12.01', 'cg': '12.01', 'ch': '12.01', 'cx': '12.01',
             'cy': '12.01', 'cu': '12.01', 'cv': '12.01', 'cz': '12.01', 'h1': '1.008', 'h2': '1.008', 'h3': '1.008',
             'h4': '1.008', 'h5': '1.008', 'ha': '1.008', 'hc': '1.008', 'hn': '1.008', 'ho': '1.008', 'hp': '1.008',
             'hs': '1.008', 'hw': '1.008', 'hx': '1.008', 'f': '19.00', 'cl': '35.45', 'br': '79.90', 'i': '126.9',
             'n': '14.01', 'n1': '14.01', 'n2': '14.01', 'n3': '14.01', 'n4': '14.01', 'na': '14.01', 'nb': '14.01',
             'nc': '14.01', 'nd': '14.01', 'ne': '14.01', 'nf': '14.01', 'nh': '14.01', 'no': '14.01', 'ni': '14.01',
             'nj': '14.01', 'nk': '14.01', 'nl': '14.01', 'nm': '14.01', 'nn': '14.01', 'np': '14.01', 'nq': '14.01',
             'o': '16.00', 'oh': '16.00', 'os': '16.00', 'op': '16.00', 'oq': '16.00', 'ow': '16.00', 'p2': '30.97',
             'p3': '30.97', 'p4': '30.97', 'p5': '30.97', 'pb': '30.97', 'pc': '30.97', 'pd': '30.97', 'pe': '30.97',
             'pf': '30.97', 'px': '30.97', 'py': '30.97', 's': '32.06', 's2': '32.06', 's4': '32.06', 's6': '32.06',
             'sh': '32.06', 'ss': '32.06', 'sp': '32.06', 'sq': '32.06', 'sx': '32.06', 'sy': '32.06', 'cs': '12.01',
             'ns': '14.01', 'nt': '14.01', 'nx': '14.01', 'ny': '14.01', 'nz': '14.01', 'n+': '14.01', 'nu': '14.01',
             'nv': '14.01', 'n7': '14.01', 'n8': '14.01', 'n9': '14.01', 'n5': '14.01', 'n6': '14.01'}


class AntechamberRunner(object):
    """
    A wrapper for AntechamberRunner software
    """

    @requires((which('parmchk') or which('parmchk2')), "Requires the binary parmchk."
                                "Install AmberTools from http://ambermd.org/#AmberTools")
    @requires(which('antechamber'), "Requires the binary antechamber."
                                    "Install AmberTools from http://ambermd.org/#AmberTools")
    @requires(which('tleap'), "Requires the binary tleap."
                              "Install AmberTools from http://ambermd.org/#AmberTools")

    def __init__(self, mols):
        """
        Args:
            mols: List of molecules
        """
        self.mols = mols
        if which('parmchk'):
            self.parmchk_version = 'parmchk'
        else:
            self.parmchk_version = 'parmchk2'

    def _run_parmchk(self, filename="mol.mol2", format="mol2", outfile_name="mol.frcmod",
                     print_improper_dihedrals="Y"):
        """
        run parmchk
        """
        command = self.parmchk_version + " -i {} -f {} -o {} -w {}".format(filename, format, outfile_name,
                                                           print_improper_dihedrals)
        exit_code = subprocess.call(shlex.split(command))
        return exit_code

    def _run_antechamber(self, filename, infile_format="gout", outfile_name="mol",
                         outfile_format="mol2", charge_method="resp", status_info=2):
        """
        run antechamber using the provided gaussian output file
        """
        command = "antechamber -i {} -fi {} -o {}.{} -fo {} -c {} -s {}".format(filename,
                                                                             infile_format,
                                                                             outfile_name,
                                                                             outfile_format,
                                                                             outfile_format,
                                                                             charge_method,
                                                                             status_info)
        # dont think 'charmm' is even an option for -fo
        # GeneralizedForceFiled tries to read in *.ac(antechamber format) file and Toplogy
        # is trying to readin *.rtf(charmm format topology) file !!! WHY?!!
        # command = 'antechamber -i ' + filename + " -fi gout -o mol -fo charmm -c resp -s 2"
        exit_code = subprocess.call(shlex.split(command))
        return exit_code

    def _run_tleap(self, mol_name='mol'):
        '''
        run tleap
        '''
        lines = []
        lines.append('source leaprc.gaff')
        lines.append('{} = loadmol2 {}.mol2'.format(mol_name,mol_name))
        lines.append('check {}'.format(mol_name))
        lines.append('loadamberparams {}.frcmod'.format(mol_name))
        # lines.append('saveoff {} {}.lib'.format(mol_name,mol_name))
        lines.append('saveamberparm {} {}.prmtop {}.inpcrd'.format(mol_name,mol_name,mol_name))
        lines.append('quit')

        text = '\n'.join(lines)

        with open('tleap.in', 'w') as file:
            file.write(text)
            file.close()

        command = 'tleap -f tleap.in'

        exit_code = subprocess.call(shlex.split(command))
        return exit_code

    def _run_tleap_existing_param(self, file_resname, ff_resname, sources = ['leaprc.ff14SB'], mol_name=None):
        if not mol_name:
            lines = []
            for source in sources:
                lines.append('source ' + source)
            lines.append('{} = {}'.format(file_resname, ff_resname))




    def _get_gaussian_ff_top_single(self, filename=None):
        """
        run antechamber using gaussian output file, then run parmchk
        to generate missing force field parameters. Store and return
        the force field and topology information in ff_mol.

        Args:
            filename: gaussian output file of the molecule

        Returns:
            Amberff namedtuple object that contains information on force field and
            topology
        """
        pass
        # scratch = tempfile.gettempdir()
        # Amberff = namedtuple("Amberff", ["force_field", "topology"])
        # with ScratchDir(scratch, copy_from_current_on_enter=True,
        #                 copy_to_current_on_exit=True) as d:
        #     # self._convert_to_pdb(mol, 'mol.pdb')
        #     # self.molname = filename.split('.')[0]
        #     self._run_antechamber(filename)
        #     self._run_parmchk()
        #     # if antechamber can't find parameters go to gaff_example.dat
        #     try:
        #         mol = Molecule.from_file('mol.rtf')
        #         print('mol.rtf file exists')
        #     except TopCorruptionException:
        #         correct_corrupted_top_files('mol.rtf', 'gaff_data.txt')
        #         top = Topology.from_file('mol.rtf')
        #         print('mol.rtf file does not exist')
        #     try:
        #         gff = ForceField.from_file('mol.frcmod')
        #     except FFCorruptionException:
        #         correct_corrupted_frcmod_files('ANTECHAMBER.FRCMOD', 'gaff_data.txt')
        #         gff = ForceField.from_file('ANTECHAMBER.FRCMOD')
            # gff.set_atom_mappings('ANTECHAMBER_AC.AC')
            # gff.read_charges()
            # decorate the molecule with the sire property "atomname"
            #mol.add_site_property("atomname", (list(gff.atom_index.values())))
        # return Amberff(gff, top)

    def get_gaussian_ff_top(self, filenames):
        """
        return a list of amber force field and topology for the list of
        gaussian output filenames corresponding to each molecule in mols list.

        Args:
            filenames (list): list of gaussian output files for each type of molecule

        Returns:
            list of Amberff namedtuples
        """
        pass
        # amber_ffs = []
        # for fname in filenames:
        #     amber_ffs.append(self._get_gaussian_ff_top_single(filename=fname))
        # return amber_ffs


def get_bond_param(amberparm,ff_label):
    """
    Reads bond force field parameters and outputs list of dictionaries in proper format for instantiating PyMatGen ForceField object.
    Removes duplicate bond parameters even when atom order is reversed.

    :param amberparm: (parmed.amber._amberparm.AmberParm) Recommended to get from parmed.load_file(filename.prmtop) method.
        The *.prmtop file should only include one molecule from running some of the methods for the Rubicon AntechamberRunner class.

    :param ff_label: (str) String that will be appended to the Amber atom type strings such that
        the same Amber atomtype from different molecules can have different force field parameters associated with them.
        Reccomended to use the molecule's name.

    :return ff_bond_types: (list)  List of Dicts containing the bond force field parameters for the molecule in the proper format
        for the PyMatGen ForceField object.
    """
    ff_bond_types = []
    for bond in amberparm.bonds:
        add_bond = True
        coeffs = [bond.type.k, bond.type.req]
        atom_types = (bond.atom1.type + ff_label,
                      bond.atom2.type + ff_label)
        for old_type in ff_bond_types:
            if coeffs == old_type['coeffs']:
                if atom_types not in old_type['types'] and atom_types[::-1] not in old_type['types']:
                    old_type['types'].append(atom_types)
                add_bond = False
                break
        if add_bond:
            ff_bond_types.append({'coeffs': coeffs, 'types': [atom_types]})
    return ff_bond_types

def get_angle_param(amberparm,ff_label):
    """
    Reads angle force field parameters and outputs list of Dicts that can be used for creating PyMatGen ForceField object.
    Similar to get_bond_param() function.

    :param amberparm: (parmed.amber._amberparm.AmberParm) Recommended to get from parmed.load_file(filename.prmtop) method.
        The *.prmtop file should only include one molecule from running some of the methods for the Rubicon AntechamberRunner class.

    :param ff_label: (str) String that will be appended to the Amber atom type strings such that
        the same Amber atomtype from different molecules can have different force field parameters associated with them.
        Reccomended to use the molecule's name.

    :return ff_bond_types: (list)  List of Dicts containing the angle force field parameters for the molecule in the proper format
        for the PyMatGen ForceField object.
    """
    ff_angle_types = []
    for angle in amberparm.angles:
        add_angle = True
        coeffs = [angle.type.k, angle.type.theteq]
        atom_types = (angle.atom1.type + ff_label,
                      angle.atom2.type + ff_label,
                      angle.atom3.type + ff_label)
        for old_type in ff_angle_types:
            if coeffs == old_type['coeffs']:
                if atom_types not in old_type['types'] and atom_types[::-1] not in old_type['types']:
                    old_type['types'].append(atom_types)
                add_angle = False
                break
        if add_angle:
            ff_angle_types.append({'coeffs': coeffs, 'types': [atom_types]})
    return ff_angle_types

def get_dihedral_param(amberparm,ff_label):
    """
    Reads dihedral force field parameters and outputs list of Dicts that can be used for creating PyMatGen ForceField objects.
    Similar to get_bond_param() function. Removes duplicate proper dihedrals even when bond order is reversed. Removes duplicate
    improper dihedrals without reversing bond order.

    :param amberparm: (parmed.amber._amberparm.AmberParm) Recommended to get from parmed.load_file(filename.prmtop) method.
        The *.prmtop file should only include one molecule from running some of the methods for the Rubicon AntechamberRunner class.

    :param ff_label: (str) String that will be appended to the Amber atom type strings such that the same Amber atomtype from
        different molecules can have different force field parameters associated with them.

    :return (ff_dihedral_types,ff_improper_types): (tuple) contains two lists of Dicts containing the proper dihedral and improper
        dihedral force field parameters for the molecule, respectively, in the proper format for the PyMatGen ForceField object.
    """
    ff_dihedral_types = []
    ff_improper_types = []
    for dihedral in amberparm.dihedrals:
        add_dihedral = True
        atom_types = (dihedral.atom1.type + ff_label,
                      dihedral.atom2.type + ff_label,
                      dihedral.atom3.type + ff_label,
                      dihedral.atom4.type + ff_label)
        if int(dihedral.type.phase) % 360 == 180:
            coeffs = [dihedral.type.phi_k, -1, dihedral.type.per]
        elif int(dihedral.type.phase) % 360 == 0:
            coeffs = [dihedral.type.phi_k, 1, dihedral.type.per]
        else:
            raise ValueError('The phase of the dihedral was a value other than 0 or 180 mod(360)')
        if not dihedral.improper:
            for old_d_type in ff_dihedral_types:
                if coeffs == old_d_type['coeffs']:
                    if atom_types not in old_d_type['types'] and atom_types[::-1] not in old_d_type['types']:
                        old_d_type['types'].append(atom_types)
                    add_dihedral = False
                    break
            if add_dihedral:
                ff_dihedral_types.append({'coeffs': coeffs, 'types': [atom_types]})
        else:
            for old_i_type in ff_improper_types:
                if coeffs == old_i_type['coeffs']:
                    if atom_types not in old_i_type['types']:
                        old_i_type['types'].append(atom_types)
                    add_dihedral = False
                    break
            if add_dihedral:
                ff_improper_types.append({'coeffs': coeffs, 'types': [atom_types]})
    return (ff_dihedral_types,ff_improper_types)

def make_ff_bonded_type_list(types_dict, mol_name=''):
    """
    Makes list for use as value in topo_coeffs input when instantiating PyMatGen ForceField object.
    Should work for bond, angle, dihedral, and improper parameters.
    :param types_dict [dict]: keys are ff parameters, values are sets of tuples of atom types involved. Intended to be obtained from OTHER_FUNCTION
    :param mol_name [str]: molecule name to use for making ff_labels (see PyMatGen Topology object)
    :return: list of dictionaries containing bonded coefficients and atom types involved
    """
    bonded_type = []
    for params in types_dict.keys():
        bonded_type.append({'coeffs': [param for param in params], 'types': []})
        for atom_types in types_dict[params]:
            atom_labels = [atom_type + mol_name for atom_type in atom_types]
            bonded_type[-1]['types'].append(tuple(atom_labels))
    return bonded_type

def get_nonbond_param(amberparm, molecule_name):
    """
    Reads mass and LJ parameters from the parmed.amberparm object and stores in the format required by the
        pymatgen.ForceField object.
    :param amberparm: (parmed.amber._amberparm.AmberParm) Recommended to get from parmed.load_file(filename.prmtop) method.
        The *.prmtop file should only include one molecule from running some of the methods for the Rubicon AntechamberRunner class.
    :param molecule_name: (str) molecule name to use for making ff_labels (see PyMatGen Topology object)
    :return: tuple of OrderedDict and list of lists. The OrderedDict contains the atom labels as keys and the masses as values.
        The list of lists contains LJ parameters that correspond to the ordering of the keys in the OrderedDict.
    """
    masses_ordered_dict = OrderedDict()
    nonbond_param_list = []
    label_list = [type + molecule_name for type in amberparm.LJ_types.keys()]
    for type in label_list:
        masses_ordered_dict[type] = float(GAFF_DICT[type[:-len(molecule_name)]])
    for type in label_list:
        index = amberparm.LJ_types[type[:-len(molecule_name)]] - 1
        nonbond_param_list.append([amberparm.LJ_depth[index],amberparm.LJ_radius[index] * 2 ** (5/6)])
    return masses_ordered_dict, nonbond_param_list

def check_partial_charge_sum(partial_charges, net_charge=0, tolerance=10**-16):
    """
    Compares the sum of the partial charges of atoms in a molecule to the net charge of the molecule.
    If the sum is not close to zero (within the tolerance), then all partial charges are shifted by the
    same amount such that the new sum of partial charges is within the tolerance of the net charge.
    :param partial_charges: [array-like] the partial charges of the atoms in a molecule
    :param net_charge: [int] the net charge of the molecule; defaults to 0
    :param tolerance: [float] the desired tolerance for the difference; defaults to 10**-16
    :return p_charges_array: [np.array] the new partial charges such that the difference is within the tolerance
    """
    p_charges_array = np.asarray(partial_charges)
    charge_difference = net_charge - np.sum(partial_charges)
    correction = charge_difference / len(partial_charges)
    if charge_difference > tolerance:
        p_charges_array += correction
    elif charge_difference < -tolerance:
        p_charges_array += correction
    return p_charges_array

def prmtop_to_python(file_name, pmg_molecule, ff_label, tolerance=10**-16):
    """
    Extracts relevant parameters from *.prmtop file containing single molecule from tleap and stores them
    in a dictionary. The partial charges are corrected such that their sum is within a tolerance of the net
    charge.
    :param file_name: [str] the filename of the *.prmtop file
    :param pmg_molecule: [pymatgen Molecule] Intended to be obtained from the same GaussianOutput object used
        to get the *.prmtop file. Make sure the net charge is set correctly.
    :param ff_label: [str] the label used to differentiate atoms with the same atomtypes, but from different
        molecules in the pymatgen.io.lammps.data ForceField object.
    :return PyParm: [dict] Contains all the relevant force field parameters for the molecule as follows:
        {'Molecule': pymatgen Molecule
        'Masses': OrderedDict([('atom_1' + ff_label, mass), ...]),
        'Nonbond': [[sigma_1, epsilon_1], ...],
        'Bonds': [{'coeffs': [k_1, r_eq_1], 'types': [('atom_a' + ff_label, 'atom_b' + ff_label), ...]}, ...],
        'Angles': [{'coeffs': [k_1, theta_eq_1], 'types': [('atom_a' + ff_label, 'atom_b' + ff_label, 'atom_c' + ff_label), ...]}, ...],
        'Dihedrals': [{'coeffs': [phi_k_1, phase_1, per_1], 'types': [('atom_a' + ff_label, 'atom_b' + ff_label, 'atom_c' + ff_label, 'atom_d' + ff_label), ...]}, ...],
        'Impropers': [{'coeffs': [phi_k_1, phase_1, per_1], 'types': [('atom_a' + ff_label, 'atom_b' + ff_label, 'atom_c' + ff_label, 'atom_d' + ff_label), ...]}, ...],
        'Charges': [charge_1, ...]
    """
    amber_parm = pmd.load_file(file_name)
    bond_parm = get_bond_param(amber_parm, ff_label)
    angle_parm = get_angle_param(amber_parm, ff_label)
    dihedral_parm, improper_parm = get_dihedral_param(amber_parm, ff_label)
    masses, nonbond_parm = get_nonbond_param(amber_parm, ff_label)
    charges = np.asarray(amber_parm.parm_data[amber_parm.charge_flag])
    corrected_charges = list(check_partial_charge_sum(charges,
                                                      pmg_molecule.charge,
                                                      tolerance))

    pyparm = {'Molecule': pmg_molecule,
              'Masses': masses,
              'Nonbond': nonbond_parm,
              'Bonds': bond_parm,
              'Angles': angle_parm,
              'Dihedrals': dihedral_parm,
              'Impropers': improper_parm,
              'Charges': corrected_charges}

    return pyparm


class PrmtopParser:
    """
    Object for parsing information necessary for LammpsDataWrapper from *.prmtop files containing a single molecule
        using the ParmEd package.
    """

    def __init__(self,prmtop_file_name,pmg_molecule,unique_molecule_name,check_partial_charges=True,tolerance=10**-16):
        self._prmtop_file_name = prmtop_file_name
        self._mol_name = unique_molecule_name
        self._molecule = pmg_molecule
        self._check_partial_charges = check_partial_charges
        self._tolerance = tolerance

        self._amberparm = pmd.load_file(self._prmtop_file_name)


    @property
    def label_type(self):
        """
        List of force field labels for each atom in the molecule.
        """
        label_list = [type + self._mol_name for type in self._amberparm.LJ_types.keys()]
        return label_list

    @property
    def masses(self):
        """
        OrderedDict of Masses in the format of OrderedDict({'atom_label_a':mass_a,...}), where 'atom_label' is a string of
            amber atomtype concatenated with the unique_molecule_label.
        """
        masses_ordered_dict = OrderedDict()
        for type in self.label_type:
            if self._mol_name:
                masses_ordered_dict[type] = float(GAFF_DICT[type[:-len(self._mol_name)].lower()])
            else:
                masses_ordered_dict[type] = float(GAFF_DICT[type.lower()])
        return masses_ordered_dict


    @property
    def lj_param(self):
        """
        List of lists for the Lennard Jones parameters in the format of [[epsilon_a, sigma_a],...].
        :return:
        """
        lj_param_list = []
        for type in self.label_type:
            if self._mol_name:
                index = self._amberparm.LJ_types[type[:-len(self._mol_name)]] - 1
            else:
                index = self._amberparm.LJ_types[type] - 1
            lj_param_list.append([self._amberparm.LJ_depth[index],self._amberparm.LJ_radius[index] * 2 ** (5/6)])
        return lj_param_list


    @property
    def bond_param(self):
        """
        List of dicts for the bond parameters in the format of [{'coeffs':coeffs_1,'types':[(i,j),...]},...].
            Automatically filters out any duplicates.
        """
        bond_param_list = []
        for bond in self._amberparm.bonds:
            add_bond = True
            coeffs = [bond.type.k, bond.type.req]
            atom_types = (bond.atom1.type + self._mol_name,
                          bond.atom2.type + self._mol_name)
            for old_type in bond_param_list:
                if coeffs == old_type['coeffs']:
                    if atom_types not in old_type['types'] and atom_types[::-1] not in old_type['types']:
                        old_type['types'].append(atom_types)
                    add_bond = False
                    break
            if add_bond:
                bond_param_list.append({'coeffs':coeffs,'types':[atom_types]})
        return bond_param_list


    @property
    def angle_param(self):
        """
        List of dicts for the angle parameters in the format of [{'coeffs':coeffs_1,'types':[(i,j,k),...]},...].
        """
        angle_param_list = []
        for angle in self._amberparm.angles:
            add_angle = True
            coeffs = [angle.type.k, angle.type.theteq]
            atom_types = (angle.atom1.type + self._mol_name,
                          angle.atom2.type + self._mol_name,
                          angle.atom3.type + self._mol_name)
            for old_type in angle_param_list:
                if coeffs == old_type['coeffs']:
                    if atom_types not in old_type['types'] and atom_types[::-1] not in old_type['types']:
                        old_type['types'].append(atom_types)
                    add_angle = False
                    break
            if add_angle:
                angle_param_list.append({'coeffs':coeffs,'types':[atom_types]})
        return angle_param_list


    @property
    def dihedral_para(self):
        """
        List of dicts for the proper dihedral parameters in the format of [{'coeffs':coeffs_1,'types':[(i,j,k,l),...]},...].
        """
        dihedral_param_list = []
        for index, dihedral in enumerate(self._amberparm.dihedrals):
            add_dihedral = True
            atom_types = (dihedral.atom1.type + self._mol_name,
                          dihedral.atom2.type + self._mol_name,
                          dihedral.atom3.type + self._mol_name,
                          dihedral.atom4.type + self._mol_name)
            if int(dihedral.type.phase) % 360 == 180:
                coeffs = [dihedral.type.phi_k, -1, dihedral.type.per]
            elif int(dihedral.type.phase) % 360 == 0:
                coeffs = [dihedral.type.phi_k, 1, dihedral.type.per]
            else:
                raise ValueError('The phase of the dihedral at index ' + str(index) + ' was a value other than 0 or 180 mod(360).')
            if not dihedral.improper:
                for old_d_type in dihedral_param_list:
                    if coeffs == old_d_type['coeffs']:
                        if atom_types not in old_d_type['types'] and atom_types[::-1] not in old_d_type['types']:
                            old_d_type['types'].append(atom_types)
                        add_dihedral = False
                        break
                if add_dihedral:
                    dihedral_param_list.append({'coeffs':coeffs,'types':[atom_types]})
        return dihedral_param_list


    @property
    def improper_param(self):
        """
        List of dicts for the improper dihedral parameters in the format of [{'coeffs':coeffs_1,'types':[(i,j,k,l),...]},...].
        """
        improper_param_list = []
        for index, dihedral in enumerate(self._amberparm.dihedrals):
            add_dihedral = True
            atom_types = (dihedral.atom1.type + self._mol_name,
                          dihedral.atom2.type + self._mol_name,
                          dihedral.atom3.type + self._mol_name,
                          dihedral.atom4.type + self._mol_name)
            if int(dihedral.type.phase) % 360 == 180:
                coeffs = [dihedral.type.phi_k, -1, dihedral.type.per]
            elif int(dihedral.type.phase) % 360 == 0:
                coeffs = [dihedral.type.phi_k, 1, dihedral.type.per]
            else:
                raise ValueError('The phase of the dihedral at index ' + str(index) + ' was a value other than 0 or 180 mod(360).')
            if dihedral.improper:
                for old_i_type in improper_param_list:
                    if coeffs == old_i_type['coeffs']:
                        if atom_types not in old_i_type['types']:
                            old_i_type['types'].append(atom_types)
                        add_dihedral = False
                        break
                if add_dihedral:
                    improper_param_list.append({'coeffs':coeffs,'types':[atom_types]})
        return improper_param_list


    @property
    def improper_topologies(self):
        """
        List of lists for the improper topologies in the format of [[index_i,index_j,index_k,index_l],...].
        :return:
        """
        impropers = []
        for dihedral in self._amberparm.dihedrals:
            if dihedral.improper:
                impropers.append(dihedral)
        improper_topology_list = []
        if impropers:
            for improper in impropers:
                atom1_index = self._amberparm.atoms.index(improper.atom1)
                atom2_index = self._amberparm.atoms.index(improper.atom2)
                atom3_index = self._amberparm.atoms.index(improper.atom3)
                atom4_index = self._amberparm.atoms.index(improper.atom4)
                improper_topology_list.append([atom1_index,atom2_index,atom3_index,atom4_index])
        else:
            improper_topology_list = None
        return improper_topology_list


    @property
    def charges(self):
        """
        np.array of partial charges. Checks to make sure that the sum of the partial charges is within a tolerance of
            the net charge of the molecule.
        :return:
        """
        charges_array = np.asarray(self._amberparm.parm_data[self._amberparm.charge_flag])
        if self._check_partial_charges:
            charge_difference = self._molecule.charge - np.sum(charges_array)
            charge_correction = charge_difference / len(charges_array)
            if charge_difference > self._tolerance:
                charges_array += charge_correction
            elif charge_difference < -self._tolerance:
                charges_array += charge_correction
        return list(charges_array)


    def to_dict(self):
        labels = [atom.type + self._mol_name for atom in self._amberparm.atoms]
        param_dict = {'Molecule':self._molecule,
                      'Labels':labels,
                      'Masses':self.masses,
                      'Nonbond':self.lj_param,
                      'Bonds':self.bond_param,
                      'Angles':self.angle_param,
                      'Dihedrals':self.dihedral_para,
                      'Impropers':self.improper_param,
                      'Improper Topologies':self.improper_topologies,
                      'Charges':self.charges}
        return param_dict