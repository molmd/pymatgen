import os
import glob
import numpy as np
import pandas as pd
from subprocess import run
from pymatgen.io.lammps.outputs import parse_lammps_dumps

'''
This script converts a single dump file (with one or more frames) into dump files (with a single frame)
and xyz files (with a single frame)
'''

cwd = os.getcwd()
dump_file_path_pattern = os.path.join(cwd,'dump.*.dump')

Dump_files = glob.glob(dump_file_path_pattern)

if len(Dump_files) == 1:
    run(['mkdir','-p','trj_files/rdf_files'])
    run(['mkdir','xyz_files'])
    Dump_file = Dump_files[0]
    wd, Dump_file_name = os.path.split(Dump_file)
    Dumps = parse_lammps_dumps(Dump_file)

    for Dump in Dumps:
        trj_name = Dump_file_name[:-4] + str(Dump.timestep) + '.lammpstrj'
        xyz_name = Dump_file_name[:-4] + 'alt.' + str(Dump.timestep) + '.xyz'
        Dump.as_txt_file(trj_name,output=True)
        Dump.as_txt_file(xyz_name,convert='xyz',output=True)
        if Dump.timestep % 500000 == 0:
            run(['cp',trj_name,'trj_files/rdf_files'])
        run(['mv',trj_name,'trj_files'])
        run(['mv',xyz_name,'xyz_files'])
