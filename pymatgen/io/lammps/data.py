# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
This module implements a core class LammpsData for generating/parsing
LAMMPS data file, and other bridging classes to build LammpsData from
molecules. This module also implements a subclass CombinedData for
merging LammpsData object, and a class LammpsDataWrapper for creating
the LammpsData object.

Only point particle styles are supported for now (atom_style in angle,
atomic, bond, charge, full and molecular only). See the pages below for
more info.

    http://lammps.sandia.gov/doc/atom_style.html
    http://lammps.sandia.gov/doc/read_data.html

"""

import itertools
import re
import warnings
from collections import OrderedDict
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from monty.dev import deprecated
from monty.json import MSONable
from monty.serialization import loadfn

from pymatgen.core import yaml
from pymatgen.core.periodic_table import Element
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.operations import SymmOp
from pymatgen.util.io_utils import clean_lines
from pymatgen.io.lammps.utils import PackmolRunner

__author__ = "Kiran Mathew, Zhi Deng, Tingzheng Hou"
__copyright__ = "Copyright 2018, The Materials Virtual Lab"
__version__ = "1.1"
__maintainer__ = "Zhi Deng, Matthew Bliss, Tingzheng Hou"
__email__ = "z4deng@eng.ucsd.edu, matthew.bliss@stonybrook.edu, " "tingzheng_hou@berkeley.edu"
__date__ = "May 29, 2021"

MODULE_DIR = Path(__file__).resolve().parent

SECTION_KEYWORDS = {
    "atom": ["Atoms", "Velocities", "Masses", "Ellipsoids", "Lines", "Triangles", "Bodies",],
    "topology": ["Bonds", "Angles", "Dihedrals", "Impropers"],
    "ff": ["Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs",],
    "class2": [
        "BondBond Coeffs",
        "BondAngle Coeffs",
        "MiddleBondTorsion Coeffs",
        "EndBondTorsion Coeffs",
        "AngleTorsion Coeffs",
        "AngleAngleTorsion Coeffs",
        "BondBond13 Coeffs",
        "AngleAngle Coeffs",
    ],
}

CLASS2_KEYWORDS = {
    "Angle Coeffs": ["BondBond Coeffs", "BondAngle Coeffs"],
    "Dihedral Coeffs": [
        "MiddleBondTorsion Coeffs",
        "EndBondTorsion Coeffs",
        "AngleTorsion Coeffs",
        "AngleAngleTorsion Coeffs",
        "BondBond13 Coeffs",
    ],
    "Improper Coeffs": ["AngleAngle Coeffs"],
}

SECTION_HEADERS = {
    "Masses": ["mass"],
    "Velocities": ["vx", "vy", "vz"],
    "Bonds": ["type", "atom1", "atom2"],
    "Angles": ["type", "atom1", "atom2", "atom3"],
    "Dihedrals": ["type", "atom1", "atom2", "atom3", "atom4"],
    "Impropers": ["type", "atom1", "atom2", "atom3", "atom4"],
}

ATOMS_HEADERS = {
    "angle": ["molecule-ID", "type", "x", "y", "z"],
    "atomic": ["type", "x", "y", "z"],
    "bond": ["molecule-ID", "type", "x", "y", "z"],
    "charge": ["type", "q", "x", "y", "z"],
    "full": ["molecule-ID", "type", "q", "x", "y", "z"],
    "molecular": ["molecule-ID", "type", "x", "y", "z"],
}

AVOGADRO = 6.02214086 * 10 ** 23  # mol^-1


class LammpsBox(MSONable):
    """
    Object for representing a simulation box in LAMMPS settings.
    """

    def __init__(self, bounds, tilt=None):
        """

        Args:
            bounds: A (3, 2) array/list of floats setting the
                boundaries of simulation box.
            tilt: A (3,) array/list of floats setting the tilt of
                simulation box. Default to None, i.e., use an
                orthogonal box.

        """
        bounds_arr = np.array(bounds)
        assert bounds_arr.shape == (3, 2,), "Expecting a (3, 2) array for bounds," " got {}".format(bounds_arr.shape)
        self.bounds = bounds_arr.tolist()
        matrix = np.diag(bounds_arr[:, 1] - bounds_arr[:, 0])

        self.tilt = None
        if tilt is not None:
            tilt_arr = np.array(tilt)
            assert tilt_arr.shape == (3,), "Expecting a (3,) array for box_tilt," " got {}".format(tilt_arr.shape)
            self.tilt = tilt_arr.tolist()
            matrix[1, 0] = tilt_arr[0]
            matrix[2, 0] = tilt_arr[1]
            matrix[2, 1] = tilt_arr[2]
        self._matrix = matrix

    def __str__(self):
        return self.get_string()

    def __repr__(self):
        return self.get_string()

    @property
    def volume(self):
        """
        Volume of simulation box.

        """
        m = self._matrix
        return np.dot(np.cross(m[0], m[1]), m[2])

    def get_string(self, significant_figures=6):
        """
        Returns the string representation of simulation box in LAMMPS
        data file format.

        Args:
            significant_figures (int): No. of significant figures to
                output for box settings. Default to 6.

        Returns:
            String representation

        """
        ph = "{:.%df}" % significant_figures
        lines = []
        for bound, d in zip(self.bounds, "xyz"):
            fillers = bound + [d] * 2
            bound_format = " ".join([ph] * 2 + [" {}lo {}hi"])
            lines.append(bound_format.format(*fillers))
        if self.tilt:
            tilt_format = " ".join([ph] * 3 + [" xy xz yz"])
            lines.append(tilt_format.format(*self.tilt))
        return "\n".join(lines)

    def get_box_shift(self, i):
        """
        Calculates the coordinate shift due to PBC.

        Args:
            i: A (n, 3) integer array containing the labels for box
            images of n entries.

        Returns:
            Coorindate shift array with the same shape of i

        """
        return np.inner(i, self._matrix.T)

    def to_lattice(self):
        """
        Converts the simulation box to a more powerful Lattice backend.
        Note that Lattice is always periodic in 3D space while a
        simulation box is not necessarily periodic in all dimensions.

        Returns:
            Lattice

        """
        return Lattice(self._matrix)


def lattice_2_lmpbox(lattice, origin=(0, 0, 0)):
    """
    Converts a lattice object to LammpsBox, and calculates the symmetry
    operation used.

    Args:
        lattice (Lattice): Input lattice.
        origin: A (3,) array/list of floats setting lower bounds of
            simulation box. Default to (0, 0, 0).

    Returns:
        LammpsBox, SymmOp

    """
    a, b, c = lattice.abc
    xlo, ylo, zlo = origin
    xhi = a + xlo
    m = lattice.matrix
    xy = np.dot(m[1], m[0] / a)
    yhi = np.sqrt(b ** 2 - xy ** 2) + ylo
    xz = np.dot(m[2], m[0] / a)
    yz = (np.dot(m[1], m[2]) - xy * xz) / (yhi - ylo)
    zhi = np.sqrt(c ** 2 - xz ** 2 - yz ** 2) + zlo
    tilt = None if lattice.is_orthogonal else [xy, xz, yz]
    rot_matrix = np.linalg.solve([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]], m)
    bounds = [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
    symmop = SymmOp.from_rotation_and_translation(rot_matrix, origin)
    return LammpsBox(bounds, tilt), symmop


class LammpsData(MSONable):
    """
    Object for representing the data in a LAMMPS data file.

    """

    def __init__(
        self, box, masses, atoms, velocities=None, force_field=None, topology=None, atom_style="full",
    ):
        """
        This is a low level constructor designed to work with parsed
        data or other bridging objects (ForceField and Topology). Not
        recommended to use directly.

        Args:
            box (LammpsBox): Simulation box.
            masses (pandas.DataFrame): DataFrame with one column
                ["mass"] for Masses section.
            atoms (pandas.DataFrame): DataFrame with multiple columns
                for Atoms section. Column names vary with atom_style.
            velocities (pandas.DataFrame): DataFrame with three columns
                ["vx", "vy", "vz"] for Velocities section. Optional
                with default to None. If not None, its index should be
                consistent with atoms.
            force_field (dict): Data for force field sections. Optional
                with default to None. Only keywords in force field and
                class 2 force field are valid keys, and each value is a
                DataFrame.
            topology (dict): Data for topology sections. Optional with
                default to None. Only keywords in topology are valid
                keys, and each value is a DataFrame.
            atom_style (str): Output atom_style. Default to "full".

        """

        if velocities is not None:
            assert len(velocities) == len(atoms), "Inconsistency found between atoms and velocities"

        if force_field:
            all_ff_kws = SECTION_KEYWORDS["ff"] + SECTION_KEYWORDS["class2"]
            force_field = {k: v for k, v in force_field.items() if k in all_ff_kws}

        if topology:
            topology = {k: v for k, v in topology.items() if k in SECTION_KEYWORDS["topology"]}

        self.box = box
        self.masses = masses
        self.atoms = atoms
        self.velocities = velocities
        self.force_field = force_field
        self.topology = topology
        self.atom_style = atom_style

    def __str__(self):
        return self.get_string()

    def __repr__(self):
        return self.get_string()

    @property
    def structure(self):
        """
        Exports a periodic structure object representing the simulation
        box.

        Return:
            Structure

        """
        masses = self.masses
        atoms = self.atoms.copy()
        if "nx" in atoms.columns:
            atoms.drop(["nx", "ny", "nz"], axis=1, inplace=True)
        atoms["molecule-ID"] = 1
        ld_copy = self.__class__(self.box, masses, atoms)
        topologies = ld_copy.disassemble()[-1]
        molecule = topologies[0].sites
        coords = molecule.cart_coords - np.array(self.box.bounds)[:, 0]
        species = molecule.species
        latt = self.box.to_lattice()
        site_properties = {}
        if "q" in atoms:
            site_properties["charge"] = atoms["q"].values
        if self.velocities is not None:
            site_properties["velocities"] = self.velocities.values
        return Structure(latt, species, coords, coords_are_cartesian=True, site_properties=site_properties,)

    def get_string(self, distance=6, velocity=8, charge=4, hybrid=True):
        """
        Returns the string representation of LammpsData, essentially
        the string to be written to a file. Support hybrid style
        coeffs read and write.

        Args:
            distance (int): No. of significant figures to output for
                box settings (bounds and tilt) and atomic coordinates.
                Default to 6.
            velocity (int): No. of significant figures to output for
                velocities. Default to 8.
            charge (int): No. of significant figures to output for
                charges. Default to 3.
            hybrid (bool): Whether to write hybrid coeffs types.
                Default to True. If the data object has no hybrid
                coeffs types and has large coeffs section, one may
                use False to speedup the process. Otherwise the
                default is recommended.

        Returns:
            String representation

        """
        file_template = """Generated by pymatgen.io.lammps.data.LammpsData

{stats}

{box}

{body}
"""
        box = self.box.get_string(distance)

        body_dict = OrderedDict()
        body_dict["Masses"] = self.masses
        types = OrderedDict()
        types["atom"] = len(self.masses)
        if self.force_field:
            all_ff_kws = SECTION_KEYWORDS["ff"] + SECTION_KEYWORDS["class2"]
            ff_kws = [k for k in all_ff_kws if k in self.force_field]
            for kw in ff_kws:
                body_dict[kw] = self.force_field[kw]
                if kw in SECTION_KEYWORDS["ff"][2:]:
                    types[kw.lower()[:-7]] = len(self.force_field[kw])

        body_dict["Atoms"] = self.atoms
        counts = OrderedDict()
        counts["atoms"] = len(self.atoms)
        if self.velocities is not None:
            body_dict["Velocities"] = self.velocities
        if self.topology:
            for kw in SECTION_KEYWORDS["topology"]:
                if kw in self.topology:
                    body_dict[kw] = self.topology[kw]
                    counts[kw.lower()] = len(self.topology[kw])

        all_stats = list(counts.values()) + list(types.values())
        stats_template = "{:>%d}  {}" % len(str(max(all_stats)))
        count_lines = [stats_template.format(v, k) for k, v in counts.items()]
        type_lines = [stats_template.format(v, k + " types") for k, v in types.items()]
        stats = "\n".join(count_lines + [""] + type_lines)

        def map_coords(q):
            return ("{:.%df}" % distance).format(q)

        def map_velos(q):
            return ("{:.%df}" % velocity).format(q)

        def map_charges(q):
            return ("{:.%df}" % charge).format(q)

        float_format = "{:.9f}".format
        float_format_2 = "{:.1f}".format
        int_format = "{:.0f}".format
        default_formatters = {
            "x": map_coords,
            "y": map_coords,
            "z": map_coords,
            "vx": map_velos,
            "vy": map_velos,
            "vz": map_velos,
            "q": map_charges,
        }
        coeffsdatatype = loadfn(str(MODULE_DIR / "CoeffsDataType.yaml"))
        coeffs = {}
        for style, types in coeffsdatatype.items():
            coeffs[style] = {}
            for type, formatter in types.items():
                coeffs[style][type] = {}
                for coeff, datatype in formatter.items():
                    if datatype == "int_format":
                        coeffs[style][type][coeff] = int_format
                    elif datatype == "float_format_2":
                        coeffs[style][type][coeff] = float_format_2
                    else:
                        coeffs[style][type][coeff] = float_format

        section_template = "{kw}\n\n{df}\n"
        parts = []
        for k, v in body_dict.items():
            index = k != "PairIJ Coeffs"
            if k in ["Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs",] and hybrid:
                listofdf = np.array_split(v, len(v.index))
                df_string = ""
                for i, df in enumerate(listofdf):
                    if isinstance(df.iloc[0]["coeff1"], str):
                        try:
                            formatters = {
                                **default_formatters,
                                **coeffs[k][df.iloc[0]["coeff1"]],
                            }
                        except KeyError:
                            formatters = default_formatters
                        line_string = df.to_string(
                            header=False, formatters=formatters, index_names=False, index=index, na_rep="",
                        )
                    else:
                        line_string = v.to_string(
                            header=False, formatters=default_formatters, index_names=False, index=index, na_rep="",
                        ).splitlines()[i]
                    df_string += line_string.replace("nan", "").rstrip() + "\n"
            else:
                df_string = v.to_string(
                    header=False, formatters=default_formatters, index_names=False, index=index, na_rep="",
                )
            parts.append(section_template.format(kw=k, df=df_string))
        body = "\n".join(parts)

        return file_template.format(stats=stats, box=box, body=body)

    def write_file(self, filename, distance=6, velocity=8, charge=4):
        """
        Writes LammpsData to file.

        Args:
            filename (str): Filename.
            distance (int): No. of significant figures to output for
                box settings (bounds and tilt) and atomic coordinates.
                Default to 6.
            velocity (int): No. of significant figures to output for
                velocities. Default to 8.
            charge (int): No. of significant figures to output for
                charges. Default to 3.

        """
        with open(filename, "w") as f:
            f.write(self.get_string(distance=distance, velocity=velocity, charge=charge))

    def disassemble(self, atom_labels=None, guess_element=True, ff_label="ff_map"):
        """
        Breaks down LammpsData to building blocks
        (LammpsBox, ForceField and a series of Topology).
        RESTRICTIONS APPLIED:

        1. No complex force field defined not just on atom
            types, where the same type or equivalent types of topology
            may have more than one set of coefficients.
        2. No intermolecular topologies (with atoms from different
            molecule-ID) since a Topology object includes data for ONE
            molecule or structure only.

        Args:
            atom_labels ([str]): List of strings (must be different
                from one another) for labelling each atom type found in
                Masses section. Default to None, where the labels are
                automaticaly added based on either element guess or
                dummy specie assignment.
            guess_element (bool): Whether to guess the element based on
                its atomic mass. Default to True, otherwise dummy
                species "Qa", "Qb", ... will be assigned to various
                atom types. The guessed or assigned elements will be
                reflected on atom labels if atom_labels is None, as
                well as on the species of molecule in each Topology.
            ff_label (str): Site property key for labeling atoms of
                different types. Default to "ff_map".

        Returns:
            LammpsBox, ForceField, [Topology]

        """
        atoms_df = self.atoms.copy()
        if "nx" in atoms_df.columns:
            atoms_df[["x", "y", "z"]] += self.box.get_box_shift(atoms_df[["nx", "ny", "nz"]].values)
        atoms_df = pd.concat([atoms_df, self.velocities], axis=1)

        mids = atoms_df.get("molecule-ID")
        if mids is None:
            unique_mids = [1]
            data_by_mols = {1: {"Atoms": atoms_df}}
        else:
            unique_mids = np.unique(mids)
            data_by_mols = {}
            for k in unique_mids:
                df = atoms_df[atoms_df["molecule-ID"] == k]
                data_by_mols[k] = {"Atoms": df}

        masses = self.masses.copy()
        masses["label"] = atom_labels
        unique_masses = np.unique(masses["mass"])
        if guess_element:
            ref_masses = [el.atomic_mass.real for el in Element]
            diff = np.abs(np.array(ref_masses) - unique_masses[:, None])
            atomic_numbers = np.argmin(diff, axis=1) + 1
            symbols = [Element.from_Z(an).symbol for an in atomic_numbers]
        else:
            symbols = ["Q%s" % a for a in map(chr, range(97, 97 + len(unique_masses)))]
        for um, s in zip(unique_masses, symbols):
            masses.loc[masses["mass"] == um, "element"] = s
        if atom_labels is None:  # add unique labels based on elements
            for el, vc in masses["element"].value_counts().iteritems():
                masses.loc[masses["element"] == el, "label"] = ["%s%d" % (el, c) for c in range(1, vc + 1)]
        assert masses["label"].nunique(dropna=False) == len(masses), "Expecting unique atom label for each type"
        mass_info = [tuple([r["label"], r["mass"]]) for _, r in masses.iterrows()]

        nonbond_coeffs, topo_coeffs = None, None
        if self.force_field:
            if "PairIJ Coeffs" in self.force_field:
                nbc = self.force_field["PairIJ Coeffs"]
                nbc = nbc.sort_values(["id1", "id2"]).drop(["id1", "id2"], axis=1)
                nonbond_coeffs = [list(t) for t in nbc.itertuples(False, None)]
            elif "Pair Coeffs" in self.force_field:
                nbc = self.force_field["Pair Coeffs"].sort_index()
                nonbond_coeffs = [list(t) for t in nbc.itertuples(False, None)]

            topo_coeffs = {k: [] for k in SECTION_KEYWORDS["ff"][2:] if k in self.force_field}
            for kw in topo_coeffs.keys():
                class2_coeffs = {
                    k: list(v.itertuples(False, None))
                    for k, v in self.force_field.items()
                    if k in CLASS2_KEYWORDS.get(kw, [])
                }
                ff_df = self.force_field[kw]
                for t in ff_df.itertuples(True, None):
                    d = {"coeffs": list(t[1:]), "types": []}
                    if class2_coeffs:
                        d.update({k: list(v[t[0] - 1]) for k, v in class2_coeffs.items()})
                    topo_coeffs[kw].append(d)

        if self.topology:

            def label_topo(t):
                return tuple(masses.loc[atoms_df.loc[t, "type"], "label"])

            for k, v in self.topology.items():
                ff_kw = k[:-1] + " Coeffs"
                for topo in v.itertuples(False, None):
                    topo_idx = topo[0] - 1
                    indices = list(topo[1:])
                    mids = atoms_df.loc[indices]["molecule-ID"].unique()
                    assert len(mids) == 1, (
                        "Do not support intermolecular topology formed " "by atoms with different molecule-IDs"
                    )
                    label = label_topo(indices)
                    topo_coeffs[ff_kw][topo_idx]["types"].append(label)
                    if data_by_mols[mids[0]].get(k):
                        data_by_mols[mids[0]][k].append(indices)
                    else:
                        data_by_mols[mids[0]][k] = [indices]

        if topo_coeffs:
            for v in topo_coeffs.values():
                for d in v:
                    d["types"] = list(set(d["types"]))

        ff = ForceField(mass_info=mass_info, nonbond_coeffs=nonbond_coeffs, topo_coeffs=topo_coeffs)

        topo_list = []
        for mid in unique_mids:
            data = data_by_mols[mid]
            atoms = data["Atoms"]
            shift = min(atoms.index)
            type_ids = atoms["type"]
            species = masses.loc[type_ids, "element"]
            labels = masses.loc[type_ids, "label"]
            coords = atoms[["x", "y", "z"]]
            m = Molecule(species.values, coords.values, site_properties={ff_label: labels.values})
            charges = atoms.get("q")
            velocities = atoms[["vx", "vy", "vz"]] if "vx" in atoms.columns else None
            topologies = {}
            for kw in SECTION_KEYWORDS["topology"]:
                if data.get(kw):
                    topologies[kw] = (np.array(data[kw]) - shift).tolist()
            topologies = None if not topologies else topologies
            topo_list.append(
                Topology(sites=m, ff_label=ff_label, charges=charges, velocities=velocities, topologies=topologies,)
            )

        return self.box, ff, topo_list

    @classmethod
    def from_file(cls, filename, atom_style="full", sort_id=False):
        """
        Constructor that parses a file.

        Args:
            filename (str): Filename to read.
            atom_style (str): Associated atom_style. Default to "full".
            sort_id (bool): Whether sort each section by id. Default to
                True.

        """
        with open(filename) as f:
            lines = f.readlines()
        kw_pattern = r"|".join(itertools.chain(*SECTION_KEYWORDS.values()))
        section_marks = [i for i, l in enumerate(lines) if re.search(kw_pattern, l)]
        parts = np.split(lines, section_marks)

        float_group = r"([0-9eE.+-]+)"
        header_pattern = dict()
        header_pattern["counts"] = r"^\s*(\d+)\s+([a-zA-Z]+)$"
        header_pattern["types"] = r"^\s*(\d+)\s+([a-zA-Z]+)\s+types$"
        header_pattern["bounds"] = r"^\s*{}$".format(r"\s+".join([float_group] * 2 + [r"([xyz])lo \3hi"]))
        header_pattern["tilt"] = r"^\s*{}$".format(r"\s+".join([float_group] * 3 + ["xy xz yz"]))

        header = {"counts": {}, "types": {}}
        bounds = {}
        for l in clean_lines(parts[0][1:]):  # skip the 1st line
            match = None
            for k, v in header_pattern.items():
                match = re.match(v, l)
                if match:
                    break
            if match and k in ["counts", "types"]:
                header[k][match.group(2)] = int(match.group(1))
            elif match and k == "bounds":
                g = match.groups()
                bounds[g[2]] = [float(i) for i in g[:2]]
            elif match and k == "tilt":
                header["tilt"] = [float(i) for i in match.groups()]
        header["bounds"] = [bounds.get(i, [-0.5, 0.5]) for i in "xyz"]
        box = LammpsBox(header["bounds"], header.get("tilt"))

        def parse_section(sec_lines):
            title_info = sec_lines[0].split("#", 1)
            kw = title_info[0].strip()
            sio = StringIO("".join(sec_lines[2:]))  # skip the 2nd line
            if kw.endswith("Coeffs") and not kw.startswith("PairIJ"):
                df_list = [
                    pd.read_csv(StringIO(line), header=None, comment="#", delim_whitespace=True)
                    for line in sec_lines[2:]
                    if line.strip()
                ]
                df = pd.concat(df_list, ignore_index=True)
                names = ["id"] + ["coeff%d" % i for i in range(1, df.shape[1])]
            else:
                df = pd.read_csv(sio, header=None, comment="#", delim_whitespace=True)
                if kw == "PairIJ Coeffs":
                    names = ["id1", "id2"] + ["coeff%d" % i for i in range(1, df.shape[1] - 1)]
                    df.index.name = None  # pylint: disable=E1101
                elif kw in SECTION_HEADERS:
                    names = ["id"] + SECTION_HEADERS[kw]
                elif kw == "Atoms":
                    names = ["id"] + ATOMS_HEADERS[atom_style]
                    if df.shape[1] == len(names):  # pylint: disable=E1101
                        pass
                    elif df.shape[1] == len(names) + 3:  # pylint: disable=E1101
                        names += ["nx", "ny", "nz"]
                    else:
                        raise ValueError("Format in Atoms section inconsistent" " with atom_style %s" % atom_style)
                else:
                    raise NotImplementedError("Parser for %s section" " not implemented" % kw)
            df.columns = names
            if sort_id:
                sort_by = "id" if kw != "PairIJ Coeffs" else ["id1", "id2"]
                df.sort_values(sort_by, inplace=True)
            if "id" in df.columns:
                df.set_index("id", drop=True, inplace=True)
                df.index.name = None
            return kw, df

        err_msg = "Bad LAMMPS data format where "
        body = {}
        seen_atoms = False
        for part in parts[1:]:
            name, section = parse_section(part)
            if name == "Atoms":
                seen_atoms = True
            if (
                name in ["Velocities"] + SECTION_KEYWORDS["topology"] and not seen_atoms
            ):  # Atoms must appear earlier than these
                raise RuntimeError(err_msg + "%s section appears before" " Atoms section" % name)
            body.update({name: section})

        err_msg += "Nos. of {} do not match between header and {} section"
        assert len(body["Masses"]) == header["types"]["atom"], err_msg.format("atom types", "Masses")
        atom_sections = ["Atoms", "Velocities"] if "Velocities" in body else ["Atoms"]
        for s in atom_sections:
            assert len(body[s]) == header["counts"]["atoms"], err_msg.format("atoms", s)
        for s in SECTION_KEYWORDS["topology"]:
            if header["counts"].get(s.lower(), 0) > 0:
                assert len(body[s]) == header["counts"][s.lower()], err_msg.format(s.lower(), s)

        items = {k.lower(): body[k] for k in ["Masses", "Atoms"]}
        items["velocities"] = body.get("Velocities")
        ff_kws = [k for k in body if k in SECTION_KEYWORDS["ff"] + SECTION_KEYWORDS["class2"]]
        items["force_field"] = {k: body[k] for k in ff_kws} if ff_kws else None
        topo_kws = [k for k in body if k in SECTION_KEYWORDS["topology"]]
        items["topology"] = {k: body[k] for k in topo_kws} if topo_kws else None
        items["atom_style"] = atom_style
        items["box"] = box
        return cls(**items)

    @classmethod
    def from_ff_and_topologies(cls, box, ff, topologies, atom_style="full"):
        """
        Constructor building LammpsData from a ForceField object and a
        list of Topology objects. Do not support intermolecular
        topologies since a Topology object includes data for ONE
        molecule or structure only.

        Args:
            box (LammpsBox): Simulation box.
            ff (ForceField): ForceField object with data for Masses and
                force field sections.
            topologies ([Topology]): List of Topology objects with data
                for Atoms, Velocities and topology sections.
            atom_style (str): Output atom_style. Default to "full".

        """
        atom_types = set.union(*[t.species for t in topologies])
        assert atom_types.issubset(ff.maps["Atoms"].keys()), "Unknown atom type found in topologies"

        items = dict(box=box, atom_style=atom_style, masses=ff.masses, force_field=ff.force_field)

        mol_ids, charges, coords, labels = [], [], [], []
        v_collector = [] if topologies[0].velocities else None
        topo_collector = {"Bonds": [], "Angles": [], "Dihedrals": [], "Impropers": []}
        topo_labels = {"Bonds": [], "Angles": [], "Dihedrals": [], "Impropers": []}
        for i, topo in enumerate(topologies):
            if topo.topologies:
                shift = len(labels)
                for k, v in topo.topologies.items():
                    topo_collector[k].append(np.array(v) + shift + 1)
                    topo_labels[k].extend([tuple(topo.type_by_sites[j] for j in t) for t in v])
            if isinstance(v_collector, list):
                v_collector.append(topo.velocities)
            mol_ids.extend([i + 1] * len(topo.sites))
            labels.extend(topo.type_by_sites)
            coords.append(topo.sites.cart_coords)
            q = [0.0] * len(topo.sites) if not topo.charges else topo.charges
            charges.extend(q)

        atoms = pd.DataFrame(np.concatenate(coords), columns=["x", "y", "z"])
        atoms["molecule-ID"] = mol_ids
        atoms["q"] = charges
        atoms["type"] = list(map(ff.maps["Atoms"].get, labels))
        atoms.index += 1
        atoms = atoms[ATOMS_HEADERS[atom_style]]

        velocities = None
        if v_collector:
            velocities = pd.DataFrame(np.concatenate(v_collector), columns=SECTION_HEADERS["Velocities"])
            velocities.index += 1

        topology = {k: None for k, v in topo_labels.items() if len(v) > 0}
        for k in topology:
            df = pd.DataFrame(np.concatenate(topo_collector[k]), columns=SECTION_HEADERS[k][1:])
            df["type"] = list(map(ff.maps[k].get, topo_labels[k]))
            if any(pd.isnull(df["type"])):  # Throw away undefined topologies
                text = "Undefined %s detected and remoded".replace("%", k.lower())
                print(text)
                # warnings.warn(text) # This warning is broken, will want to fix in future
                df.dropna(subset=["type"], inplace=True)
                df.reset_index(drop=True, inplace=True)
                df = df.astype("int")
            df.index += 1
            topology[k] = df[SECTION_HEADERS[k]]
        topology = {k: v for k, v in topology.items() if not v.empty}

        items.update({"atoms": atoms, "velocities": velocities, "topology": topology})
        return cls(**items)

    @classmethod
    def from_structure(cls, structure, ff_elements=None, atom_style="charge", is_sort=False):
        """
        Simple constructor building LammpsData from a structure without
        force field parameters and topologies.

        Args:
            structure (Structure): Input structure.
            ff_elements ([str]): List of strings of elements that must
                be present due to force field settings but not
                necessarily in the structure. Default to None.
            atom_style (str): Choose between "atomic" (neutral) and
                "charge" (charged). Default to "charge".
            is_sort (bool): whether to sort sites

        """
        if is_sort:
            s = structure.get_sorted_structure()
        else:
            s = structure.copy()
        box, symmop = lattice_2_lmpbox(s.lattice)
        coords = symmop.operate_multi(s.cart_coords)
        site_properties = s.site_properties
        if "velocities" in site_properties:
            velos = np.array(s.site_properties["velocities"])
            rot = SymmOp.from_rotation_and_translation(symmop.rotation_matrix)
            rot_velos = rot.operate_multi(velos)
            site_properties.update({"velocities": rot_velos})
        boxed_s = Structure(
            box.to_lattice(), s.species, coords, site_properties=site_properties, coords_are_cartesian=True,
        )

        symbols = list(s.symbol_set)
        if ff_elements:
            symbols.extend(ff_elements)
        elements = sorted(Element(el) for el in set(symbols))
        mass_info = [tuple([i.symbol] * 2) for i in elements]
        ff = ForceField(mass_info)
        topo = Topology(boxed_s)
        return cls.from_ff_and_topologies(box=box, ff=ff, topologies=[topo], atom_style=atom_style)

    @classmethod
    def from_dict(cls, d):
        """
        Constructor that reads in a dictionary.

        Args:
            d (dict): Dictionary to read.
        """

        def decode_df(s):
            return pd.read_json(s, orient="split")

        items = dict()
        items["box"] = LammpsBox.from_dict(d["box"])
        items["masses"] = decode_df(d["masses"])
        items["atoms"] = decode_df(d["atoms"])
        items["atom_style"] = d["atom_style"]

        velocities = d["velocities"]
        if velocities:
            velocities = decode_df(velocities)
        items["velocities"] = velocities
        force_field = d["force_field"]
        if force_field:
            force_field = {k: decode_df(v) for k, v in force_field.items()}
        items["force_field"] = force_field
        topology = d["topology"]
        if topology:
            topology = {k: decode_df(v) for k, v in topology.items()}
        items["topology"] = topology
        return cls(**items)

    def as_dict(self):
        """
        Returns the LammpsData as a dict.

        """

        def encode_df(df):
            return df.to_json(orient="split")

        d = dict()
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["box"] = self.box.as_dict()
        d["masses"] = encode_df(self.masses)
        d["atoms"] = encode_df(self.atoms)
        d["atom_style"] = self.atom_style

        d["velocities"] = None if self.velocities is None else encode_df(self.velocities)
        d["force_field"] = None if not self.force_field else {k: encode_df(v) for k, v in self.force_field.items()}
        d["topology"] = None if not self.topology else {k: encode_df(v) for k, v in self.topology.items()}
        return d


class Topology(MSONable):
    """
    Class carrying most data in Atoms, Velocities and molecular
    topology sections for ONE SINGLE Molecule or Structure
    object, or a plain list of Sites.

    """

    def __init__(self, sites, ff_label=None, charges=None, velocities=None, topologies=None):
        """

        Args:
            sites ([Site] or SiteCollection): A group of sites in a
                list or as a Molecule/Structure.
            ff_label (str): Site property key for labeling atoms of
                different types. Default to None, i.e., use
                site.species_string.
            charges ([q, ...]): Charge of each site in a (n,)
                array/list, where n is the No. of sites. Default to
                None, i.e., search site property for charges.
            velocities ([[vx, vy, vz], ...]): Velocity of each site in
                a (n, 3) array/list, where n is the No. of sites.
                Default to None, i.e., search site property for
                velocities.
            topologies (dict): Bonds, angles, dihedrals and improper
                dihedrals defined by site indices. Default to None,
                i.e., no additional topology. All four valid keys
                listed below are optional.
                {
                    "Bonds": [[i, j], ...],
                    "Angles": [[i, j, k], ...],
                    "Dihedrals": [[i, j, k, l], ...],
                    "Impropers": [[i, j, k, l], ...]
                }

        """
        if not isinstance(sites, (Molecule, Structure)):
            sites = Molecule.from_sites(sites)

        if ff_label:
            type_by_sites = sites.site_properties.get(ff_label)
        else:
            type_by_sites = [site.specie.symbol for site in sites]
        # search for site property if not override
        if charges is None:
            charges = sites.site_properties.get("charge")
        if velocities is None:
            velocities = sites.site_properties.get("velocities")
        # validate shape
        if charges is not None:
            charge_arr = np.array(charges)
            assert charge_arr.shape == (len(sites),), "Wrong format for charges"
            charges = charge_arr.tolist()
        if velocities is not None:
            velocities_arr = np.array(velocities)
            assert velocities_arr.shape == (len(sites), 3,), "Wrong format for velocities"
            velocities = velocities_arr.tolist()

        if topologies:
            topologies = {k: v for k, v in topologies.items() if k in SECTION_KEYWORDS["topology"]}

        self.sites = sites
        self.ff_label = ff_label
        self.charges = charges
        self.velocities = velocities
        self.topologies = topologies
        self.type_by_sites = type_by_sites
        self.species = set(type_by_sites)

    @classmethod
    def from_bonding(cls, molecule, bond=True, angle=True, dihedral=True, tol=0.1, **kwargs):
        """
        Another constructor that creates an instance from a molecule.
        Covalent bonds and other bond-based topologies (angles and
        dihedrals) can be automatically determined. Cannot be used for
        non bond-based topologies, e.g., improper dihedrals.

        Args:
            molecule (Molecule): Input molecule.
            bond (bool): Whether find bonds. If set to False, angle and
                dihedral searching will be skipped. Default to True.
            angle (bool): Whether find angles. Default to True.
            dihedral (bool): Whether find dihedrals. Default to True.
            tol (float): Bond distance tolerance. Default to 0.1.
                Not recommended to alter.
            **kwargs: Other kwargs supported by Topology.

        """
        real_bonds = molecule.get_covalent_bonds(tol=tol)
        bond_list = [list(map(molecule.index, [b.site1, b.site2])) for b in real_bonds]
        if not all((bond, bond_list)):
            # do not search for others if not searching for bonds or no bonds
            return cls(sites=molecule, **kwargs)

        angle_list, dihedral_list = [], []
        dests, freq = np.unique(bond_list, return_counts=True)
        hubs = dests[np.where(freq > 1)].tolist()
        bond_arr = np.array(bond_list)
        if len(hubs) > 0:
            hub_spokes = {}
            for hub in hubs:
                ix = np.any(np.isin(bond_arr, hub), axis=1)
                bonds = np.unique(bond_arr[ix]).tolist()
                bonds.remove(hub)
                hub_spokes[hub] = bonds
        # skip angle or dihedral searching if too few bonds or hubs
        dihedral = False if len(bond_list) < 3 or len(hubs) < 2 else dihedral
        angle = False if len(bond_list) < 2 or len(hubs) < 1 else angle

        if angle:
            for k, v in hub_spokes.items():
                angle_list.extend([[i, k, j] for i, j in itertools.combinations(v, 2)])
        if dihedral:
            hub_cons = bond_arr[np.all(np.isin(bond_arr, hubs), axis=1)]
            for i, j in hub_cons.tolist():
                ks = [k for k in hub_spokes[i] if k != j]
                ls = [l for l in hub_spokes[j] if l != i]
                dihedral_list.extend([[k, i, j, l] for k, l in itertools.product(ks, ls) if k != l])

        topologies = {
            k: v for k, v in zip(SECTION_KEYWORDS["topology"][:3], [bond_list, angle_list, dihedral_list]) if len(v) > 0
        }
        topologies = None if len(topologies) == 0 else topologies
        return cls(sites=molecule, topologies=topologies, **kwargs)


class ForceField(MSONable):
    """
    Class carrying most data in Masses and force field sections.

    Attributes:
        masses (pandas.DataFrame): DataFrame for Masses section.
        force_field (dict): Force field section keywords (keys) and
            data (values) as DataFrames.
        maps (dict): Dict for labeling atoms and topologies.

    """

    @staticmethod
    def _is_valid(df):
        return not pd.isnull(df).values.any()

    def __init__(self, mass_info, nonbond_coeffs=None, topo_coeffs=None, check_duplicates=True):
        """

        Args:
            mass_into (list): List of atomic mass info. Elements,
                strings (symbols) and floats are all acceptable for the
                values, with the first two converted to the atomic mass
                of an element. It is recommended to use
                OrderedDict.items() to prevent key duplications.
                [("C", 12.01), ("H", Element("H")), ("O", "O"), ...]
            nonbond_coeffs [coeffs]: List of pair or pairij
                coefficients, of which the sequence must be sorted
                according to the species in mass_dict. Pair or PairIJ
                determined by the length of list. Optional with default
                to None.
            topo_coeffs (dict): Dict with force field coefficients for
                molecular topologies. Optional with default
                to None. All four valid keys listed below are optional.
                Each value is a list of dicts with non optional keys
                "coeffs" and "types", and related class2 force field
                keywords as optional keys.
                {
                    "Bond Coeffs":
                        [{"coeffs": [coeff],
                          "types": [("C", "C"), ...]}, ...],
                    "Angle Coeffs":
                        [{"coeffs": [coeff],
                          "BondBond Coeffs": [coeff],
                          "types": [("H", "C", "H"), ...]}, ...],
                    "Dihedral Coeffs":
                        [{"coeffs": [coeff],
                          "BondBond13 Coeffs": [coeff],
                          "types": [("H", "C", "C", "H"), ...]}, ...],
                    "Improper Coeffs":
                        [{"coeffs": [coeff],
                          "AngleAngle Coeffs": [coeff],
                          "types": [("H", "C", "C", "H"), ...]}, ...],
                }
                Topology of same type or equivalent types (e.g.,
                ("C", "H") and ("H", "C") bonds) are NOT ALLOWED to
                be defined MORE THAN ONCE with DIFFERENT coefficients.

        """
        self._check_duplicates = check_duplicates

        def map_mass(v):
            return (
                v.atomic_mass.real
                if isinstance(v, Element)
                else Element(v).atomic_mass.real
                if isinstance(v, str)
                else v
            )

        index, masses, self.mass_info, atoms_map = [], [], [], {}
        for i, m in enumerate(mass_info):
            index.append(i + 1)
            mass = map_mass(m[1])
            masses.append(mass)
            self.mass_info.append((m[0], mass))
            atoms_map[m[0]] = i + 1
        self.masses = pd.DataFrame({"mass": masses}, index=index)
        self.maps = {"Atoms": atoms_map}

        ff_dfs = {}

        self.nonbond_coeffs = nonbond_coeffs
        if self.nonbond_coeffs:
            ff_dfs.update(self._process_nonbond())

        self.topo_coeffs = topo_coeffs
        if self.topo_coeffs:
            self.topo_coeffs = {k: v for k, v in self.topo_coeffs.items() if k in SECTION_KEYWORDS["ff"][2:]}
            for k in self.topo_coeffs.keys():
                coeffs, mapper = self._process_topo(k)
                ff_dfs.update(coeffs)
                self.maps.update(mapper)

        self.force_field = None if len(ff_dfs) == 0 else ff_dfs

    def _process_nonbond(self):
        pair_df = pd.DataFrame(self.nonbond_coeffs)
        assert self._is_valid(pair_df), "Invalid nonbond coefficients with rows varying in length"
        npair, ncoeff = pair_df.shape
        pair_df.columns = ["coeff%d" % i for i in range(1, ncoeff + 1)]
        nm = len(self.mass_info)
        ncomb = int(nm * (nm + 1) / 2)
        if npair == nm:
            kw = "Pair Coeffs"
            pair_df.index = range(1, nm + 1)
        elif npair == ncomb:
            kw = "PairIJ Coeffs"
            ids = list(itertools.combinations_with_replacement(range(1, nm + 1), 2))
            id_df = pd.DataFrame(ids, columns=["id1", "id2"])
            pair_df = pd.concat([id_df, pair_df], axis=1)
        else:
            raise ValueError(
                "Expecting {} Pair Coeffs or "
                "{} PairIJ Coeffs for {} atom types,"
                " got {}".format(nm, ncomb, nm, npair)
            )
        return {kw: pair_df}

    def _process_topo(self, kw):
        def find_eq_types(label, section):
            if section.startswith("Improper"):
                label_arr = np.array(label)
                seqs = [[0, 1, 2, 3], [0, 2, 1, 3], [3, 1, 2, 0], [3, 2, 1, 0]]
                return [tuple(label_arr[s]) for s in seqs]
            return [label] + [label[::-1]]

        main_data, distinct_types = [], []
        class2_data = {k: [] for k in self.topo_coeffs[kw][0].keys() if k in CLASS2_KEYWORDS.get(kw, [])}
        for i, d in enumerate(self.topo_coeffs[kw]):
            main_data.append(d["coeffs"])
            distinct_types.append(d["types"])
            for k in class2_data.keys():
                class2_data[k].append(d[k])
        if self._check_duplicates:
            distinct_types = [set(itertools.chain(*[find_eq_types(t, kw) for t in dt])) for dt in distinct_types]
            type_counts = sum([len(dt) for dt in distinct_types])
            type_union = set.union(*distinct_types)
            assert len(type_union) == type_counts, "Duplicated items found " "under different coefficients in %s" % kw
        else:
            distinct_types = [list(itertools.chain(*[find_eq_types(t, kw) for t in dt])) for dt in distinct_types]
        atoms = set(np.ravel(list(itertools.chain(*distinct_types))))
        assert atoms.issubset(self.maps["Atoms"].keys()), "Undefined atom type found in %s" % kw
        mapper = {}
        if self._check_duplicates:
            for i, dt in enumerate(distinct_types):
                for t in dt:
                    mapper[t] = i + 1
        else:
            for i, dt in enumerate(distinct_types):
                for t in dt:
                    mapper[tuple(t)] = i + 1

        def process_data(data):
            df = pd.DataFrame(data)
            assert self._is_valid(df), "Invalid coefficients with rows varying in length"
            n, c = df.shape
            df.columns = ["coeff%d" % i for i in range(1, c + 1)]
            df.index = range(1, n + 1)
            return df

        all_data = {kw: process_data(main_data)}
        if class2_data:
            all_data.update({k: process_data(v) for k, v in class2_data.items()})
        return all_data, {kw[:-7] + "s": mapper}

    def to_file(self, filename):
        """
        Saves object to a file in YAML format.

        Args:
            filename (str): Filename.

        """
        d = {
            "mass_info": self.mass_info,
            "nonbond_coeffs": self.nonbond_coeffs,
            "topo_coeffs": self.topo_coeffs,
        }
        with open(filename, "w") as f:
            yaml.dump(d, f)

    @classmethod
    def from_file(cls, filename):
        """
        Constructor that reads in a file in YAML format.

        Args:
            filename (str): Filename.

        """
        with open(filename, "r") as f:
            d = yaml.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d):
        """
        Constructor that reads in a dictionary.

        Args:
            d (dict): Dictionary to read.
        """
        d["mass_info"] = [tuple(m) for m in d["mass_info"]]
        if d.get("topo_coeffs"):
            for v in d["topo_coeffs"].values():
                for c in v:
                    c["types"] = [tuple(t) for t in c["types"]]
        return cls(d["mass_info"], d["nonbond_coeffs"], d["topo_coeffs"])


class CombinedData(LammpsData):
    """
    Object for a collective set of data for a series of LAMMPS data file.
    velocities not yet implementd.
    """

    def __init__(
        self, list_of_molecules, list_of_names, list_of_numbers, coordinates, atom_style="full",
    ):
        """
        Args:
            list_of_molecules: A list of LammpsData objects of a chemical cluster.
                 Each LammpsData object (cluster) may contain one or more molecule ID.
            list_of_names: A list of name (string) for each cluster. The characters in each name are
                restricted to word characters ([a-zA-Z0-9_]). If names with any non-word characters
                are passed in, the special characters will be substituted by '_'.
            list_of_numbers: A list of Integer for counts of each molecule
                coordinates (pandas.DataFrame): DataFrame with with four
                columns ["atom", "x", "y", "z"] for coordinates of atoms.
            atom_style (str): Output atom_style. Default to "full".

        """

        max_xyz = coordinates[["x", "y", "z"]].max().max()
        min_xyz = coordinates[["x", "y", "z"]].min().min()
        self.box = LammpsBox(np.array(3 * [[min_xyz - 0.5, max_xyz + 0.5]]))
        self.atom_style = atom_style
        self.n = sum(list_of_numbers)
        self.names = list()
        for name in list_of_names:
            self.names.append("_".join(re.findall(r"\w+", name)))
        self.mols = list_of_molecules
        self.nums = list_of_numbers
        self.masses = pd.concat([mol.masses.copy() for mol in self.mols], ignore_index=True)
        self.masses.index += 1
        all_ff_kws = SECTION_KEYWORDS["ff"] + SECTION_KEYWORDS["class2"]
        appeared_kws = {k for mol in self.mols if mol.force_field is not None for k in mol.force_field}
        ff_kws = [k for k in all_ff_kws if k in appeared_kws]
        self.force_field = {}
        for kw in ff_kws:
            self.force_field[kw] = pd.concat(
                [mol.force_field[kw].copy() for mol in self.mols if kw in mol.force_field], ignore_index=True,
            )
            self.force_field[kw].index += 1
        if not bool(self.force_field):
            self.force_field = None

        self.atoms = pd.DataFrame()
        mol_count = 0
        type_count = 0
        self.mols_per_data = list()
        for i, mol in enumerate(self.mols):
            atoms_df = mol.atoms.copy()
            atoms_df["molecule-ID"] += mol_count
            atoms_df["type"] += type_count
            mols_in_data = len(atoms_df["molecule-ID"].unique())
            self.mols_per_data.append(mols_in_data)
            for j in range(self.nums[i]):
                self.atoms = self.atoms.append(atoms_df, ignore_index=True)
                atoms_df["molecule-ID"] += mols_in_data
            type_count += len(mol.masses)
            mol_count += self.nums[i] * mols_in_data
        self.atoms.index += 1
        assert len(self.atoms) == len(coordinates), "Wrong number of coordinates."
        self.atoms.update(coordinates)

        self.velocities = None
        assert self.mols[0].velocities is None, "Velocities not supported"

        self.topology = {}
        atom_count = 0
        count = {"Bonds": 0, "Angles": 0, "Dihedrals": 0, "Impropers": 0}
        for i, mol in enumerate(self.mols):
            for kw in SECTION_KEYWORDS["topology"]:
                if bool(mol.topology) and kw in mol.topology:
                    if kw not in self.topology:
                        self.topology[kw] = pd.DataFrame()
                    topo_df = mol.topology[kw].copy()
                    topo_df["type"] += count[kw]
                    for col in topo_df.columns[1:]:
                        topo_df[col] += atom_count
                    for j in range(self.nums[i]):
                        self.topology[kw] = self.topology[kw].append(topo_df, ignore_index=True)
                        for col in topo_df.columns[1:]:
                            topo_df[col] += len(mol.atoms)
                    count[kw] += len(mol.force_field[kw[:-1] + " Coeffs"])
            atom_count += len(mol.atoms) * self.nums[i]
        for kw in SECTION_KEYWORDS["topology"]:
            if kw in self.topology:
                self.topology[kw].index += 1
        if not bool(self.topology):
            self.topology = None

    @classmethod
    def parse_xyz(cls, filename):
        """
        load xyz file generated from packmol (for those who find it hard to install openbabel)

        Returns:
            pandas.DataFrame

        """
        with open(filename) as f:
            lines = f.readlines()

        sio = StringIO("".join(lines[2:]))  # skip the 2nd line
        df = pd.read_csv(sio, header=None, comment="#", delim_whitespace=True, names=["atom", "x", "y", "z"],)
        df.index += 1
        return df

    @classmethod
    def from_files(cls, coordinate_file, list_of_numbers, *filenames):
        """
        Constructor that parse a series of data file.

        Args:
            coordinate_file (str): The filename of xyz coordinates.
            list_of_numbers (list): A list of numbers specifying counts for each
                clusters parsed from files.
            filenames (str): A series of LAMMPS data filenames in string format.
        """
        names = []
        mols = []
        styles = []
        coordinates = cls.parse_xyz(filename=coordinate_file)
        for i in range(0, len(filenames)):
            exec("cluster%d = LammpsData.from_file(filenames[i])" % (i + 1))
            names.append("cluster%d" % (i + 1))
            mols.append(eval("cluster%d" % (i + 1)))
            styles.append(eval("cluster%d" % (i + 1)).atom_style)
        style = set(styles)
        assert len(style) == 1, "Files have different atom styles."
        return cls.from_lammpsdata(mols, names, list_of_numbers, coordinates, style.pop())

    @classmethod
    def from_lammpsdata(cls, mols, names, list_of_numbers, coordinates, atom_style=None):
        """
        Constructor that can infer atom_style.
        The input LammpsData objects are used non-destructively.

        Args:
            mols: a list of LammpsData of a chemical cluster.Each LammpsData object (cluster)
                may contain one or more molecule ID.
            names: a list of name for each cluster.
            list_of_numbers: a list of Integer for counts of each molecule
                coordinates (pandas.DataFrame): DataFrame with with four
                columns ["atom", "x", "y", "z"] for coordinates of atoms.
            atom_style (str): Output atom_style. Default to "full".
        """
        styles = []
        for mol in mols:
            styles.append(mol.atom_style)
        style = set(styles)
        assert len(style) == 1, "Data have different atom_style."
        style_return = style.pop()
        if atom_style:
            assert atom_style == style_return, "Data have different atom_style as specified."
        return cls(mols, names, list_of_numbers, coordinates, style_return)

    def get_string(self, distance=6, velocity=8, charge=4):
        """
        Returns the string representation of CombinedData, essentially
        the string to be written to a file. Combination info is included
        as a comment. For single molecule ID data, the info format is:
            num name
        For data with multiple molecule ID, the format is:
            num(mols_per_data) name

        Args:
            distance (int): No. of significant figures to output for
                box settings (bounds and tilt) and atomic coordinates.
                Default to 6.
            velocity (int): No. of significant figures to output for
                velocities. Default to 8.
            charge (int): No. of significant figures to output for
                charges. Default to 3.

        Returns:
            String representation
        """
        lines = LammpsData.get_string(self, distance, velocity, charge).splitlines()
        info = "# " + " + ".join(
            (str(a) + " " + b) if c == 1 else (str(a) + "(" + str(c) + ") " + b)
            for a, b, c in zip(self.nums, self.names, self.mols_per_data)
        )
        lines.insert(1, info)
        return "\n".join(lines)

    def as_lammpsdata(self):
        """
        Convert a CombinedData object to a LammpsData object.

        """
        items = dict()
        items["box"] = self.box
        items["masses"] = self.masses
        items["atoms"] = self.atoms
        items["atom_style"] = self.atom_style
        items["velocities"] = self.velocities
        items["force_field"] = self.force_field
        items["topology"] = self.topology
        return LammpsData(**items)


def split_by_mol(n_atoms, n_mols, mix, low_ind=0, site_prop=None, molecules=[]):
    """
    Breaks a Molecule object w/ many molecules into many Molecule objects w/ a single
    molecule each. Not intended to be used directly, but part of LammpsDataWrapper class.

    :param n_atoms (Int): The number of atoms in the molecule, from num_sites()
        method on Molecule w/ a single molecule
    :param n_mols (Int): The number of molecules in the system,
        from param_list['number'] property on PackmolRunner object (or input)
    :param mix (Molecule): The object w/ all molecules in the system,
        from PackmolRunner.run() output object
    :param low_ind (Int): The index of the first atom in the molecule, defaults to 0
    :param site_prop (Dict): For adding site_property to each single Molecule object,
        may want to include 'ff_label' and 'charge', defaults to None
    :param molecules (List): The initial list of single Molecule objects, defaults to []
    :return: molecules (List): The current list of single Molecule objects
    :return: low_ind (Int): The current index
    """
    upp_ind = low_ind + n_atoms
    for i in range(n_mols):
        sites = mix.sites[low_ind:upp_ind]
        mol = Molecule.from_sites(sites)
        if site_prop:
            for key in site_prop.keys():
                mol.add_site_property(key, site_prop[key])
        molecules.append(mol)
        low_ind += n_atoms
        upp_ind += n_atoms
    return molecules, low_ind


def split_by_mult_mol(packmol_mols, packmol_params, mix, site_props=None):
    """
    Breaks a Molecule object w/ multiple kinds of molecules into many Molecule objects
        containing a single molecule each. Intended to be used as part of
        LammpsDataWrapper class

    :param packmol_mols (list): list of Molecule objects, from PackmolRunner
        input or from PackmolRunner.mols
    :param packmol_params (list): list of dicts, from PackmolRunner input or
        from PackmolRunner.param_list
    :param mix (Molecule): Molecule containing many molecules, from PackmolRunner.run()
    :param site_props (list): list of dicts, for adding site_properties to molecules,
        should contain 'ff_label' and 'charge', defaults to None
    :return: molecules (list): list of Molecule objects containing a single molecule each.
    """
    assert len(packmol_mols) == len(packmol_params), \
        "Length of input lists should be the same"

    ind = 0
    molecules = []

    if site_props is not None:
        assert len(site_props) == len(
            packmol_mols
        ), "Length of existing site property list should be the same as input lists"

        for i, mol in enumerate(packmol_mols):
            molecules, ind = split_by_mol(
                mol.num_sites,
                packmol_params[i]["number"],
                mix,
                low_ind=ind,
                site_prop=site_props[i],
                molecules=molecules,
            )
        return molecules
    else:
        for i, mol in enumerate(packmol_mols):
            molecules, ind = split_by_mol(
                mol.num_sites,
                packmol_params[i]["number"],
                mix,
                low_ind=ind,
                molecules=molecules
            )
        return molecules


def calc_num_mols(box_volume, solute_list, solvent_list):
    """
    Calculates the number of molecules for each molecular species in the system.
        This is important to obtain the input information for the PackmolRunner
        class in pymatgen.io.lammps.utils. The desired output is the second element
        of the output tuple
    :param box_volume: [float] the volume of the system box in cubic angstroms.
        Intended to be obtained from LammpsBox.volume
    :param solute_list: [list] contains Dictionaries with the following as keys:
        ['Initial Molarity', 'Final Molarity', 'Density', 'Molar Weight'] for
        each molecular species that is a solute (ie has a defined molarity).
        'Initial Molarity' and 'Final Molarity' will be equal unless some of
        the solute is transformed into another molecular species upon mixing
        (eg 1 M DHPS in 4 M NaOH (aq) will become 1 M DHPS^-3, 4 M Na^+,
        1 M OH^- after mixing; the 'Initial Molarity' of OH^- is 4 and the
        'Final Molarity' of OH^- is 1).
    :param solvent_list: [list] the same as solute list, except there is no
        'Initial Molarity' or 'Final Molarity' in the keys. Currently only
        supports a solvent with one type of molecule (ie the length of this
        parameter should be 1).
    :return: [tuple] Contains two elements. Both elements are lists of the number
        of molecules in the system, with the first elements corresponding to
        those in the solute_list, and the last elements corresponding to those in
        the solvent list. The first list is based on the 'Initial Molarity' of
        each solute molecule only. The purpose of this is mainly for checking
        whether the correct number of molecules has been transferred to another
        component. The second list is the desired output, which is based on the
        'Final Molarity' of each solute. For the second list, the number of solvent
        molecules is increased by the difference between the number of solute
        molecules from the 'Initial Molarity' and 'Final Molarity'.
    """
    # assert len(solvent_list) == 1
    if len(solvent_list) != 1:
        raise ValueError("The length of the solvent list must be 1.")
    volumes_initial = np.zeros(len(solute_list) + len(solvent_list))
    volumes_final = volumes_initial.copy()
    nmols_initial = volumes_initial.copy()
    nmols_final = volumes_initial.copy()
    for i, solute in enumerate(solute_list):
        nmols_initial[i] = int(round(solute["Initial Molarity"] *
                                     AVOGADRO * 10 ** 3 * 10 ** -30 * box_volume))
        nmols_final[i] = int(round(solute["Final Molarity"] *
                                   AVOGADRO * 10 ** 3 * 10 ** -30 * box_volume))
        volumes_initial[i] = (
            nmols_initial[i] / AVOGADRO * solute["Molar Weight"] / solute["Density"]
            * 100 ** -3 * 10 ** 30
        )
        volumes_final[i] = nmols_final[i] / AVOGADRO * solute["Molar Weight"] / \
                           solute["Density"] * 100 ** -3 * 10 ** 30
    if solute_list:
        volumes_initial[-1] = box_volume - np.sum(volumes_initial)
    else:
        volumes_initial[0] = box_volume
    extra_solvent = np.sum(np.subtract(nmols_initial, nmols_final))
    # assert extra_solvent >= 0
    if solute_list:
        nmols_initial[-1] = int(
            round(
                solvent_list[0]["Density"]
                * 100 ** 3
                * 10 ** -30
                / solvent_list[0]["Molar Weight"]
                * AVOGADRO
                * volumes_initial[-1]
            )
        )
        nmols_final[-1] = int(
            round(
                solvent_list[0]["Density"]
                * 100 ** 3
                * 10 ** -30
                / solvent_list[0]["Molar Weight"]
                * AVOGADRO
                * volumes_initial[-1]
                + extra_solvent
            )
        )
    else:
        nmols_initial[0] = int(
            round(
                solvent_list[0]["Density"]
                * 100 ** 3
                * 10 ** -30
                / solvent_list[0]["Molar Weight"]
                * AVOGADRO
                * volumes_initial[0]
            )
        )
        nmols_final[0] = int(
            round(
                solvent_list[0]["Density"]
                * 100 ** 3
                * 10 ** -30
                / solvent_list[0]["Molar Weight"]
                * AVOGADRO
                * volumes_initial[0]
                + extra_solvent
            )
        )
    return nmols_initial, nmols_final


def check_sys_charge(n_mols, volume, solute_list, solvent_list, max_change=5,
                     lambdas=[1, 1, 1]):
    """
    Checks that the number of molecules for each species yields a net system charge
        of 0. If that is not the case, the function determines what the number of
        molecules for each species should be in order to make the net charge 0 while
        making sure that the new number of molecules is sufficiently close to the set
        system based on concentration, not needlessly changing the numbers, and
        ratio of solutes to the first one in the solute list.
    :param n_mols: [list] Contains the number of molecules for each species in the
        system. Intended to be the output of Calc_Num_Mols().
    :param volume: [float or int] the volume of the system in cubic angstroms.
    :param solute_list: [list of dicts] list of system mixture dictionaries for
        the solutes.
    :param solvent_list: [list of dicts] list of system mixture dictionaries
        for the solvents.
    :param max_change: [int] The maximum value each number of molecules can be
        altered. Defaults to 5.
    :param lambdas: [list] The weights for each component of the loss function.
        The order corresponds to the following components: concentration,
        number of changes, and ratio. The first value may need to be increased.
        Defaults to [1, 1, 1].
    :return: [tuple] Contains two elements. The first is the number of molecules as
        an np.array of shape (1, x), where x is the number of species in the system.
        The second element is the net system charge.
    """
    solute_charges = [i["Charge"] for i in solute_list]
    solvent_charges = [i["Charge"] for i in solvent_list]
    charges = solute_charges + solvent_charges

    sys_charge = np.sum(np.multiply(n_mols, charges))

    if sys_charge == 0:
        return n_mols, sys_charge

    # Check all arrangements of changing number of molecules for each species
    # up to max_change.
    # Create matrix whose rows contain all such arrangements: design_mat.
    raw_design_vals = np.reshape(np.arange(-1 * max_change, 1 + max_change, 1),
                                 (-1, 2 * max_change + 1))
    full_fact_dict = dict(zip(np.arange(0, len(charges), 1),
                              np.repeat(raw_design_vals, len(charges), axis=0)))
    raw_design_mat = build.full_fact(full_fact_dict).values
    nmol_mat = np.repeat(np.reshape(np.asarray(n_mols),
                                    (-1, len(n_mols))),
                         np.shape(raw_design_mat)[0], axis=0)
    design_mat = np.add(raw_design_mat, nmol_mat)

    # calculate sum of squared errors relating to concentration for each arrangement
    set_conc = np.reshape(np.asarray([i["Final Molarity"] for i in solute_list]),
                          (-1, len(solute_list)))
    set_conc_mat = np.repeat(set_conc, np.shape(design_mat)[0], axis=0)
    conc_const = 1 / AVOGADRO / volume * (10 ** 30) / (10 ** 3)
    design_conc_mat = np.multiply(design_mat, conc_const)
    conc_err_mat = np.subtract(design_conc_mat[:, : len(solute_list)], set_conc_mat)
    conc_sq_err_mat = np.square(conc_err_mat)
    conc_sum_sq_err = np.multiply(np.sum(conc_sq_err_mat, axis=1), lambdas[0])

    # calculate penalty for the size of change of molecules for each arrangement
    size_sq_mat = np.square(raw_design_mat)
    size_sum_sq = np.multiply(np.sum(size_sq_mat, axis=1), lambdas[1])

    # calculate sum of squared errors relating to the ratio to first solute for each
    # arrangement
    set_ratio = np.divide(set_conc, set_conc[0, 0])
    set_ratio_mat = np.repeat(set_ratio, np.shape(design_mat)[0], axis=0)
    design_ratio_mat = np.divide(
        design_mat[:, : len(solute_list)],
        np.repeat(np.reshape(design_mat[:, 0], (len(design_mat), 1)),
                  len(solute_list), axis=1),
    )
    ratio_err_mat = np.subtract(design_ratio_mat, set_ratio_mat)
    ratio_sq_err_mat = np.square(ratio_err_mat)
    ratio_sum_sq_err = np.multiply(np.sum(ratio_sq_err_mat, axis=1), lambdas[2])

    # calculate loss function for each arangement
    loss_func = np.add(np.add(conc_sum_sq_err, size_sum_sq), ratio_sum_sq_err)

    # get indices of arrangements that yield 0 net charge
    mol_charges = np.repeat(np.reshape(np.asarray(charges), (-1, len(charges))),
                            np.shape(design_mat)[0], axis=0)
    design_charges_mat = np.multiply(design_mat, mol_charges)
    design_charges = np.sum(design_charges_mat, axis=1)
    zero_charge_ind = np.flatnonzero(design_charges == 0)
    if len(zero_charge_ind) == 0:
        return n_mols, sys_charge

    # take minimum value of loss function of arrangement that yields 0 net charge
    loss_func_zero = loss_func[zero_charge_ind]
    loss_func_zero_min_ind = np.flatnonzero(loss_func_zero == loss_func_zero.min())
    design_mat_zero = design_mat[zero_charge_ind]
    n_mols_opt = design_mat_zero[loss_func_zero_min_ind][0]
    return n_mols_opt, 0


class LammpsDataWrapper:
    """
    Object for wrapping LammpsData object in pymatgen.io.lammps.data
    """

    def __init__(
        self,
        system_force_fields,
        system_mixture_data,
        box_data,
        mixture_data_type="concentration",
        origin=[0.0, 0.0, 0.0],
        seed=150,
        packmolrunner_inputs={
            "input_file": "pack.inp",
            "tolerance": 2.0,
            "filetype": "xyz",
            "control_params": {"maxit": 20, "nloop": 600, "seed": 150},
            "auto_box": False,
            "output_file": "packed.xyz",
            "bin": "packmol",
            "copy_to_current_on_exit": False,
            "site_property": None,
        },
        length_increase=0.5,
        check_ff_duplicates=True,
        box_data_type="cubic",
    ):
        """
        Low level constructor designed to work with lists of dictionaries that should
        be able to be obtained from databases. Works for cubic boxes only using
        real coordinates.
        :param system_force_fields: [dict] Contains force field information using
            the following format:
            { unique_molecule_name: {
                'Molecule': pymatgen.Molecule,
                'Labels': [atom_a, ...]
                'Masses': [OrderedDict({species_1: mass_1, ...})],
                'Nonbond': [[...], ...],
                'Bonds': [{'coeffs': [...], 'types': [(i, j), ...]}, ...],
                'Angles': [{'coeffs': [...], 'types': [(i, j, k), ...]}, ...],
                'Dihedrals': [{'coeffs': [...], 'types': [(i, j, k, l), ...]}, ...],
                'Impropers': [{'coeffs': [...], 'types': [(i, j, k, l), ...]}, ...],
                'Improper Topologies': [[a, b, c, d],...]
                'Charges': [atom_a, ...]
            }, ...}
        :param system_mixture_data: [dict] Format depends on mixture_data_type input.
            For mixture_data_type = 'concentration', this parameter contains molarity,
            density, and molar weights of solutes and solvents using the
            following format:
            {
                'Solutes': {unique_molecule_name: {
                                'Initial Molarity': molarity_1i,
                                'Final Molarity': molarity_1f,
                                'Density': density_1,
                                'Molar Weight': molar_weight_1
                            }, ...},
                'Solvents': {unique_molecule_name: {
                                'Initial Molarity': molarity_1i,
                                'Final Molarity': molarity_1f,
                                'Density': density_1,
                                'Molar Weight': molar_weight_1
                            }, ...}
            }
            For mixture_data_type = 'number of molecules', this parameter contains
            the number of molecules for each species in the system in the following
            format:
            {
                unique_molecule_name: n_mols,
                ...
            }
        :param mixture_data_type: [str] controls the format of the system_mixture_data
            parameter. Currently supports values of 'concentration' and
            'number of molecules'. Defaults to "concentration".
        :param cube_length: [float] length of system box in angstroms.
        :param origin: [list] Optional. Change if the minimum xyz coordinates for
            desired box are not [0,0,0].
        :param seed: [int] Optional. Sets the seed for running packmol.
        :param packmolrunner_inputs: [dict] Optional. Parameters for PackmolRunner
            in pymatgen.io.lammps.utils
        """
        self._ff_list = system_force_fields
        self._concentration_data = False
        self._number_of_molecules_data = False

        if mixture_data_type == "concentration":
            self._concentration_data = True
            if "Solutes" in system_mixture_data.keys():
                self._solutes = system_mixture_data["Solutes"]

            else:
                self._solutes = {}
            if "Solvents" in system_mixture_data.keys():
                self._solvents = system_mixture_data["Solvents"]
            else:
                self._solvents = {}
        elif mixture_data_type == "number of molecules":
            self._number_of_molecules_data = True
            self._n_mol_dict = system_mixture_data

        if box_data_type == "cubic":
            self.length = box_data
            self._initial_lammps_box = LammpsBox([[0.0, box_data],
                                                  [0.0, box_data],
                                                  [0.0, box_data]])
        elif box_data_type == "rectangular":
            self._initial_lammps_box = LammpsBox(box_data)
        elif box_data_type == "LammpsBox":
            self._initial_lammps_box = box_data
        self._origin = origin

        packmolrunner_inputs["control_params"]["seed"] = seed
        self._packmolrunner_inputs = packmolrunner_inputs

        self.xyz_high = [bound[1] for bound in self._initial_lammps_box.as_dict()["bounds"]]
        self.xyz_low = [bound[0] for bound in self._initial_lammps_box.as_dict()["bounds"]]

        self._length_increase = length_increase
        self._check_ff_duplicates = check_ff_duplicates

    @property
    def sorted_mol_names(self):
        """
        Sorts molecules from most to least number of atoms
        :return molecule_name_list: [list] Contains the unique_molecule_names
        """
        molecule_name_list = list(self._ff_list.keys())
        molecule_natoms_list = [len(self._ff_list[name]["Molecule"]) for
                                name in molecule_name_list]
        molecule_name_list.sort(key=dict(zip(molecule_name_list,
                                             molecule_natoms_list)).get,
                                reverse=True)
        return molecule_name_list

    @property
    def nmol_dict(self):
        if self._concentration_data:
            # Convert _solute and _solvent info to lists, preserving order with
            # respect to SortedNames
            solute_names = [name for name in self.sorted_mol_names
                            if name in self._solutes.keys()]
            solvent_names = [name for name in self.sorted_mol_names
                             if name in self._solvents.keys()]
            solute_list = [
                dict(
                    itertools.chain(
                        self._solutes[name].items(),
                        {"Charge": self._ff_list[name]["Molecule"].charge}.items()
                    )
                )
                for name in solute_names
            ]
            solvent_list = [
                dict(
                    itertools.chain(
                        self._solvents[name].items(),
                        {"Charge": self._ff_list[name]["Molecule"].charge}.items()
                    )
                )
                for name in solvent_names
            ]

            # Set the number of molecules based on molarity, density, and molar weight
            # as values of Dict with keys of the unique_molecule_name
            nmolecules_initial, nmolecules_final = calc_num_mols(
                self._initial_lammps_box.volume, solute_list, solvent_list
            )

            # Check that the net charge of the system is zero
            nmols_checked, charge_checked = check_sys_charge(
                nmolecules_final,
                self._initial_lammps_box.volume,
                solute_list,
                solvent_list
            )
            nmol_dict = dict(zip(solute_names + solvent_names, nmols_checked))

        elif self._number_of_molecules_data:
            nmol_dict = self._n_mol_dict

        return nmol_dict

    @property
    def packmol_param_list(self):
        """
        Prepares the param_list input for PackmolRunner in pymatgen.io.lammps.utils.
        Assumes that all molecules are put in the same cube. Preserves the order of
        SortedNames; the molecule with the most atoms is first.
        :return packmol_params: [list] Info about number of atoms and box coordinates
        for packmol in Dicts for each  molecule.
        """
        nmol_dict = self.nmol_dict

        # # Create list of min and max xyz coords
        self.xyz_low = [bound[0] for bound in self._initial_lammps_box.as_dict()['bounds']]
        self.xyz_high = [bound[1] for bound in self._initial_lammps_box.as_dict()['bounds']]
        box_xyz = self.xyz_low + self.xyz_high

        # make packmol input list preserving the order of SortedNames
        packmol_params = [{"number": int(nmol_dict[name]), "inside box": box_xyz}
                          for name in self.sorted_mol_names]
        return packmol_params

    @property
    def packmol_mol_list(self):
        """
        Prepares the mols input for PackmolRunner in pymatgen.io.lammps.utils.
        Preserves the order of SortedNames.
        :return packmol_mols: [list] Contains pymatgen.Molecule objects for each molecule.
        """
        packmol_mols = [self._ff_list[name]["Molecule"] for name in self.sorted_mol_names]
        return packmol_mols

    @property
    def num_molecules(self):
        """
        Contains the number of molecules in the system in the order based on the
        SortedNames list.
        :return number_of_molecules: [list] Contains integers for each name in SortedNames list.
        """
        number_of_molecules = [species["number"] for species in self.packmol_param_list]
        return number_of_molecules

    @property
    def get_force_field(self):
        # Prepare OrderedDict for Masses
        masses_ordered_dict = OrderedDict()
        for species in self.sorted_mol_names:
            temp_old_list = list(masses_ordered_dict.items())
            temp_append_list = list(self._ff_list[species]["Masses"].items())
            if temp_old_list:
                masses_ordered_dict = OrderedDict(temp_old_list + temp_append_list)
            else:
                masses_ordered_dict = OrderedDict(temp_append_list)

        # Prepare List of Nonbonded Parameters
        nonbonded_list_of_lists = [self._ff_list[species]["Nonbond"] for
                                   species in self.sorted_mol_names]
        nonbonded_param_list = list(itertools.chain(*nonbonded_list_of_lists))

        # Prepare List of Bond Parameters
        bond_list_of_lists = [self._ff_list[species]["Bonds"] for
                              species in self.sorted_mol_names]
        bond_param_list = list(itertools.chain(*bond_list_of_lists))

        # Prepare List of Angle Parameters
        angle_list_of_lists = [
            self._ff_list[species]["Angles"] for species in self.sorted_mol_names
            if self._ff_list[species]["Angles"]
        ]
        angle_param_list = list(itertools.chain(*angle_list_of_lists))

        # Prepare List of Dihedral Parameters
        dihedral_list_of_lists = [
            self._ff_list[species]["Dihedrals"]
            for species in self.sorted_mol_names
            if self._ff_list[species]["Dihedrals"]
        ]
        dihedral_param_list = list(itertools.chain(*dihedral_list_of_lists))

        # Prepare List of Improper Parameters
        improper_list_of_lists = [
            self._ff_list[species]["Impropers"]
            for species in self.sorted_mol_names
            if self._ff_list[species]["Impropers"]
        ]
        improper_param_list = list(itertools.chain(*improper_list_of_lists))

        system_bonded_params = {}
        if bond_param_list:
            system_bonded_params["Bond Coeffs"] = bond_param_list
        if angle_param_list:
            system_bonded_params["Angle Coeffs"] = angle_param_list
        if dihedral_param_list:
            system_bonded_params["Dihedral Coeffs"] = dihedral_param_list
        if improper_param_list:
            system_bonded_params["Improper Coeffs"] = improper_param_list
        system_force_field_params = ForceField(
            masses_ordered_dict.items(),
            nonbonded_param_list,
            system_bonded_params,
            check_duplicates=self._check_ff_duplicates,
        )
        return system_force_field_params

    def _run_packmol(self, verbose=False):
        """
        Wrapper for PackmolRunner class in pymatgen.io.lammps.utils
        :param verbose: [bool] Optional. If True, prints additional information
        :return System_molecule_obj: [pmg.Molecule] The packed system in the form of
        pymatgen Molecule.
        """
        if verbose:
            print("The seed is:", self._packmolrunner_inputs["control_params"]["seed"])

        packmol_runner_obj = PackmolRunner(
            self.packmol_mol_list,
            self.packmol_param_list,
            input_file=self._packmolrunner_inputs["input_file"],
            tolerance=self._packmolrunner_inputs["tolerance"],
            filetype=self._packmolrunner_inputs["filetype"],
            control_params=self._packmolrunner_inputs["control_params"],
            auto_box=self._packmolrunner_inputs["auto_box"],
            output_file=self._packmolrunner_inputs["output_file"],
            bin=self._packmolrunner_inputs["bin"],
        )

        if verbose:
            print("Running Packmol:")
        System_molecule_obj = packmol_runner_obj.run(
            site_property=self._packmolrunner_inputs["site_property"]
        )
        if verbose:
            print("Packmol finished!")
        return System_molecule_obj

    def _get_topologies(self, system_molecule, verbose=False):
        """
        Get list of pmg.Topology objects from pymatgen.io.lammps.data based on the
        output from the _run_packmol() method.
        :param system_molecule: [Molecule] Output from _run_packmol()
        :return system_individual_topologies: [list] list of pmg.Topology objects
        """
        system_site_props = [
            {"ff_label": self._ff_list[name]["Labels"],
             "charge": list(self._ff_list[name]["Charges"])}
            for name in self.sorted_mol_names
        ]
        system_individual_molecules = split_by_mult_mol(
            self.packmol_mol_list,
            self.packmol_param_list,
            system_molecule,
            site_props=system_site_props
        )
        bins = np.subtract(np.cumsum(self.num_molecules), 1)
        system_individual_topologies = []
        if verbose:
            print("Starting the topology list creation loop.")
        for index, molecule in enumerate(system_individual_molecules):
            bin_result = np.digitize(index, bins, right=True)
            if verbose and index % 1000 == 0:
                print("Current loop number:", index)
                print(bin_result)
            input_topology = Topology.from_bonding(molecule)
            bin_result = np.digitize(index, bins, right=True)
            if self._ff_list[self.sorted_mol_names[bin_result]]["Improper Topologies"]:
                input_topology.topologies["Impropers"] = \
                    self._ff_list[self.sorted_mol_names[bin_result]]["Improper Topologies"]
            output_topology = Topology(
                sites=molecule,
                ff_label="ff_label",
                charges=None,
                velocities=None,
                topologies=input_topology.topologies
            )
            system_individual_topologies.append(output_topology)

        return system_individual_topologies

    def _get_lammps_box(self, system_molecule):
        """
        Get pmg.LammpsBox object from pymatgen.io.lammps.data based on the output from
        the _run_packmol() method.
        :param system_molecule: [Molecule] Output from _run_packmol()
        :return Mix_lmpbox: [pmg.LammpsBox] Object representing the simulation box
        """
        final_xyz_low = np.subtract(self.xyz_low, np.ones(3) * self._length_increase * 0.5)
        final_xyz_high = np.add(self.xyz_high, np.ones(3) * self._length_increase * 0.5)
        final_bounds = np.asarray([final_xyz_low, final_xyz_high]).transpose()

        mix_lmpbox = LammpsBox(final_bounds, self._initial_lammps_box.as_dict()["tilt"])
        return mix_lmpbox

    def build_lammps_data(self, atom_style="full"):
        system_molecule = self._run_packmol()
        system_topologies = self._get_topologies(system_molecule)
        system_lammps_box = self._get_lammps_box(system_molecule)
        system_lammps_data = LammpsData.from_ff_and_topologies(
            system_lammps_box,
            self.get_force_field,
            system_topologies,
            atom_style=atom_style
        )
        return system_lammps_data

    @property
    def ff_list(self):
        return self._ff_list


@deprecated(
    LammpsData.from_structure,
    "structure_2_lmpdata has been deprecated in favor of LammpsData.from_structure"
)
def structure_2_lmpdata(structure, ff_elements=None, atom_style="charge", is_sort=False):
    """
    Converts a structure to a LammpsData object with no force field
    parameters and topologies.

    Args:
        structure (Structure): Input structure.
        ff_elements ([str]): List of strings of elements that must be
            present due to force field settings but not necessarily in
            the structure. Default to None.
        atom_style (str): Choose between "atomic" (neutral) and
            "charge" (charged). Default to "charge".
        is_sort (bool): whether to sort the structure sites
    Returns:
        LammpsData

    """
    if is_sort:
        s = structure.get_sorted_structure()
    else:
        s = structure.copy()

    a, b, c = s.lattice.abc
    m = s.lattice.matrix
    xhi = a
    xy = np.dot(m[1], m[0] / xhi)
    yhi = np.sqrt(b ** 2 - xy ** 2)
    xz = np.dot(m[2], m[0] / xhi)
    yz = (np.dot(m[1], m[2]) - xy * xz) / yhi
    zhi = np.sqrt(c ** 2 - xz ** 2 - yz ** 2)
    box_bounds = [[0.0, xhi], [0.0, yhi], [0.0, zhi]]
    box_tilt = [xy, xz, yz]
    box_tilt = None if not any(box_tilt) else box_tilt
    box = LammpsBox(box_bounds, box_tilt)
    new_latt = Lattice([[xhi, 0, 0], [xy, yhi, 0], [xz, yz, zhi]])
    s.lattice = new_latt

    symbols = list(s.symbol_set)
    if ff_elements:
        symbols.extend(ff_elements)
    elements = sorted(Element(el) for el in set(symbols))
    mass_info = [tuple([i.symbol] * 2) for i in elements]
    ff = ForceField(mass_info)
    topo = Topology(s)
    return LammpsData.from_ff_and_topologies(box=box,
                                             ff=ff,
                                             topologies=[topo],
                                             atom_style=atom_style)
