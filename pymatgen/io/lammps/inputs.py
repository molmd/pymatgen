# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
This module implements methods for writing LAMMPS input files.
"""


import os
import re
import shutil
import warnings
from string import Template
from tabulate import tabulate

from monty.io import zopen
from monty.json import MSONable
from monty.os.path import zpath

from pymatgen.util.string import str_delimited
from pymatgen.io.lammps.data import LammpsData

__author__ = "Kiran Mathew, Brandon Wood, Zhi Deng"
__copyright__ = "Copyright 2018, The Materials Virtual Lab"
__version__ = "1.0"
__maintainer__ = "Zhi Deng"
__email__ = "z4deng@eng.ucsd.edu"
__date__ = "Aug 1, 2018"

COMMON_DUP_KEYWORDS = ["group", "fix", "compute", "unfix"]


class LammpsRun(MSONable):
    """
    Examples for various simple LAMMPS runs with given simulation box,
    force field and a few more settings. Experience LAMMPS users should
    consider using write_lammps_inputs method with more sophisticated
    templates.

    """

    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

    def __init__(self, script_template, settings, data, script_filename):
        """
        Base constructor.

        Args:
            script_template (str): String template for input script
                with placeholders. The format for placeholders has to
                be '$variable_name', e.g., '$temperature'
            settings (dict): Contains values to be written to the
                placeholders, e.g., {'temperature': 1}.
            data (LammpsData or str): Data file as a LammpsData
                instance or path to an existing data file. Default to
                None, i.e., no data file supplied. Useful only when
                read_data cmd is in the script.
            script_filename (str): Filename for the input script.

        """
        self.script_template = script_template
        self.settings = settings
        self.data = data
        self.script_filename = script_filename

    def write_inputs(self, output_dir, **kwargs):
        """
        Writes all input files (input script, and data if needed).
        Other supporting files are not handled at this moment.

        Args:
            output_dir (str): Directory to output the input files.
            **kwargs: kwargs supported by LammpsData.write_file.

        """
        write_lammps_inputs(
            output_dir=output_dir,
            script_template=self.script_template,
            settings=self.settings,
            data=self.data,
            script_filename=self.script_filename,
            **kwargs,
        )

    @classmethod
    def md(cls, data, force_field, temperature, nsteps, other_settings=None):
        r"""
        Example for a simple MD run based on template md.txt.

        Args:
            data (LammpsData or str): Data file as a LammpsData
                instance or path to an existing data file.
            force_field (str): Combined force field related cmds. For
                example, 'pair_style eam\npair_coeff * * Cu_u3.eam'.
            temperature (float): Simulation temperature.
            nsteps (int): No. of steps to run.
            other_settings (dict): other settings to be filled into
                placeholders.

        """
        template_path = os.path.join(cls.template_dir, "md.txt")
        with open(template_path) as f:
            script_template = f.read()
        settings = other_settings.copy() if other_settings is not None else {}
        settings.update({"force_field": force_field, "temperature": temperature, "nsteps": nsteps})
        script_filename = "in.md"
        return cls(
            script_template=script_template,
            settings=settings,
            data=data,
            script_filename=script_filename,
        )


def write_lammps_inputs(
    output_dir,
    script_template,
    settings=None,
    data=None,
    script_filename="in.lammps",
    make_dir_if_not_present=True,
    **kwargs,
):
    """
    Writes input files for a LAMMPS run. Input script is constructed
    from a str template with placeholders to be filled by custom
    settings. Data file is either written from a LammpsData
    instance or copied from an existing file if read_data cmd is
    inspected in the input script. Other supporting files are not
    handled at the moment.

    Args:
        output_dir (str): Directory to output the input files.
        script_template (str): String template for input script with
            placeholders. The format for placeholders has to be
            '$variable_name', e.g., '$temperature'
        settings (dict): Contains values to be written to the
            placeholders, e.g., {'temperature': 1}. Default to None.
        data (LammpsData or str): Data file as a LammpsData instance or
            path to an existing data file. Default to None, i.e., no
            data file supplied. Useful only when read_data cmd is in
            the script.
        script_filename (str): Filename for the input script.
        make_dir_if_not_present (bool): Set to True if you want the
            directory (and the whole path) to be created if it is not
            present.
        **kwargs: kwargs supported by LammpsData.write_file.

    Examples:
        >>> eam_template = '''units           metal
        ... atom_style      atomic
        ...
        ... lattice         fcc 3.615
        ... region          box block 0 20 0 20 0 20
        ... create_box      1 box
        ... create_atoms    1 box
        ...
        ... pair_style      eam
        ... pair_coeff      1 1 Cu_u3.eam
        ...
        ... velocity        all create $temperature 376847 loop geom
        ...
        ... neighbor        1.0 bin
        ... neigh_modify    delay 5 every 1
        ...
        ... fix             1 all nvt temp $temperature $temperature 0.1
        ...
        ... timestep        0.005
        ...
        ... run             $nsteps'''
        >>> write_lammps_inputs('.', eam_template, settings={'temperature': 1600.0, 'nsteps': 100})
        >>> with open('in.lammps') as f:
        ...     script = f.read()
        ...
        >>> print(script)
        units           metal
        atom_style      atomic

        lattice         fcc 3.615
        region          box block 0 20 0 20 0 20
        create_box      1 box
        create_atoms    1 box

        pair_style      eam
        pair_coeff      1 1 Cu_u3.eam

        velocity        all create 1600.0 376847 loop geom

        neighbor        1.0 bin
        neigh_modify    delay 5 every 1

        fix             1 all nvt temp 1600.0 1600.0 0.1

        timestep        0.005

        run             100


    """
    variables = {} if settings is None else settings
    template = Template(script_template)
    input_script = template.safe_substitute(**variables)
    if make_dir_if_not_present and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, script_filename), "w") as f:
        f.write(input_script)
    read_data = re.search(r"read_data\s+(.*)\n", input_script)
    if read_data:
        data_filename = read_data.group(1).split()[0]
        if isinstance(data, LammpsData):
            data.write_file(os.path.join(output_dir, data_filename), **kwargs)
        elif isinstance(data, str) and os.path.exists(data):
            shutil.copyfile(data, os.path.join(output_dir, data_filename))
        else:
            warnings.warn("No data file supplied. Skip writing %s."
                          % data_filename)
            
class LammpsInput(dict, MSONable):
    """
    Object to read a LAMMPS input file to a python object
    """
    def __init__(self, params=None):
        """"""
        super().__init__()
        
        if params:
            self.update(params)

    def __setitem__(self, key, val):
        """
        Add parameter-val pair to LammpsInput. Cleans the parameter and val
        by stripping leading and trailing white spaces.
        """
        super().__setitem__(
            key.strip(),
            val.strip(),
        )

    def as_dict(self):
        """
        :return: MSONable dict.
        """
        d = dict(self)
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        return d

    @classmethod
    def from_dict(cls, d):
        """
        :param d: Dict representation.
        :return: lammps_input
        """
        return LammpsInput({k: v for k, v in d.items() if k not in ("@module", "@class")})

    def get_string(self, sort_keys=False, pretty=False):
        """
        Returns a string representation of the input file. The reason why this
        method is different from the __str__ method is to provide options for
        pretty printing.

        Args:
            sort_keys (bool): Set to True to sort the INCAR parameters
                alphabetically. Defaults to False.
            pretty (bool): Set to True for pretty aligned output. Defaults
                to False.
        """
        keys = self.keys()
        lines = []

        for k in keys:
            if isinstance(self[k], list):
                lines.append([k, " ".join([str(i) for i in self[k]])])
            else:
                lines.append([k, self[k]])

        if pretty:
            return str(tabulate([[l[0], "", l[1]] for l in lines],
                                tablefmt="plain"))
        return str_delimited(lines, None, "\t") + "\n"

    def __str__(self):
        return self.get_string(sort_keys=False, pretty=True)

    def write_file(self, filename):
        """
        Write LAMMPS input to a file.

        Args:
            filename (str): filename to write to.
        """
        with zopen(filename, "wt") as f:
            f.write(self.__str__())

    @staticmethod
    def from_directory(input_dir, optional_files=None):
        """
        Read in a LAMMPS input file from a directory. Note that only the
        standard input file is read unless optional_filenames is specified.

        Args:
            input_dir (str): Directory to read LAMMPS input from.
            optional_files (dict): Optional files to read in as well as a
                dict of {filename: Object type}. Object type must have a
                static method from_file.
        """
        sub_d = {}
        fullzpath = zpath(os.path.join(input_dir, "lammps_input"))
        return LammpsInput.from_file(fullzpath)

    @staticmethod
    def from_file(filename):
        """
        Reads an LammpsInput object from a file.

        Args:
            filename (str): Filename for file

        Returns:
            LammpsInput object
        """
        with zopen(filename, "rt") as f:
            return LammpsInput.from_string(f.read())

    @staticmethod
    def from_string(string):
        """
        Reads an LammpsInput object from a string.

        Args:
            string (str): LammpsInput string

        Returns:
            LammpsInput object
        """
        lines = list(string.splitlines())
        params = {}
        counter = 1
        for line in lines:
            line = line.replace("\t", " ")
            m = line.strip().split(" ")
            m = [el.strip() for el in m if el != ""]
            if m:
                if "#" in m:
                    key = " ".join(m)
                    val = " "
                elif m[0] in COMMON_DUP_KEYWORDS:
                    key = " ".join(m[:2])
                    val = " ".join(m[2:])
                else:
                    key = m[0]
                    val = " ".join(m[1:])
            else:
                key = " " * counter
                val = " "
                counter += 1
            params[key] = val
        print(params)
        return LammpsInput(params)
