import argparse
import pandas as pd
import numpy as np
from pymatgen.io.lammps.outputs import parse_lammps_dumps
from sklearn.linear_model import LinearRegression

'''
This script parses data from dump files, computes MSD, and writes msd.dat csv file. Assumes real units.
'''
# # Command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--file_pattern','-fp',help='set file pattern before wildcard character (*), if no input, full file pattern will be *.lammpstrj')
parser.add_argument('--timestep','-ts',help='set # of fs per timestep, similar to lammps control file argument.',type=float)
parser.add_argument('--cutoff_time','-ct',help='time after which data will be considered for self-diffusivity calculation. Defaults to 500 ps')

args = parser.parse_args()

if args.file_pattern:
    file_pattern = args.file_pattern + '*'
else:
    file_pattern = '*.lammpstrj'

if args.timestep:
    fs_per_step = args.timestep
else:
    fs_per_step = 1

if args.cutoff_time:
    cutoff_time = args.cutoff_time
else:
    cutoff_time = 500

print('File pattern used: ' + file_pattern)
print('Timestep used: ' + str(fs_per_step) + ' fs')
print('Cutoff time used: ' + str(cutoff_time) + ' ps')
# # Convert dump files to list of LammpsDump objects
# file_pattern = 'trj_files/msd_files/dump.0.5_dhps_2.5_naoh_spce.nvt_production.*.lammpstrj'
Dumps = list(parse_lammps_dumps(file_pattern))

print('Number of frames found: ' + str(len(Dumps)))

# # ps per timestep conversion factor
ps_per_step = fs_per_step / 1000

# # Instantiate data object. Time is in ps, msd is in square Angstroms
Data = pd.DataFrame(columns=['Time','msd_x','msd_y','msd_z','msd'])

for i, dump in enumerate(Dumps):
    print('Processing frame number ' + str(i))
    # # Sorting for atom id
    dump.data = dump.data.sort_values(by=['id'])
    dump.data.reset_index(inplace=True)

    # # Box dimensions
    box_x_length = dump.box.bounds[0][1] - dump.box.bounds[0][0]
    box_y_length = dump.box.bounds[1][1] - dump.box.bounds[1][0]
    box_z_length = dump.box.bounds[2][1] - dump.box.bounds[2][0]

    # # Ensuring access to unwrapped coordinates, may want to add scaled coordinates in future ('sx', 'sxu', etc.)
    if 'xu' and 'yu' and 'zu' not in dump.data.columns:
        assert('x' and 'y' and 'z' in dump.data.columns), 'Missing coordinates'
        assert('ix' and 'iy' and 'iz' in dump.data.columns), 'Missing unwrapped coordinates and box location for converting wrapped coordinates.'
        dump.data['xu'] = dump.data['x'].add(dump.data['ix'].multiply(box_x_length))
        dump.data['yu'] = dump.data['y'].add(dump.data['iy'].multiply(box_x_length))
        dump.data['zu'] = dump.data['z'].add(dump.data['iz'].multiply(box_x_length))

    # # Making square displacement data
    dump.data[['dx2','dy2','dz2']] = dump.data[['xu','yu','zu']].subtract(Dumps[0].data[['xu','yu','zu']],axis=1).pow(2)
    dump.data['disp2'] = dump.data[['dx2','dy2','dz2']].sum(axis=1)

    # print(dump.data.head(5))

    # # Computing mean square displacements and adding them to Data object
    time = dump.timestep * ps_per_step
    msd_x = dump.data['dx2'].mean()
    msd_y = dump.data['dy2'].mean()
    msd_z = dump.data['dz2'].mean()
    msd = dump.data['disp2'].mean()
    # print(time)
    # print(msd_x)
    # print(msd_y)
    # print(msd_z)
    # print(msd)
    Data = Data.append({'Time':time,'msd_x':msd_x,'msd_y':msd_y,'msd_z':msd_z,'msd':msd},ignore_index=True)

# print(Data.head(11))

# # Write to csv file
out_file_name = 'msd.dat'
Data.to_csv(out_file_name,index=False)
print('MSD data written to: ' + out_file_name)

# Test_data = pd.read_csv('msd.dat')
# print(Test_data.head(11))

# # Linear regression
t = np.asarray(Data['Time'].drop(Data[Data.Time < cutoff_time].index)).reshape((-1,1))
f_Dx = np.asarray(Data['msd_x'].drop(Data[Data.Time < cutoff_time].index))
f_Dy = np.asarray(Data['msd_y'].drop(Data[Data.Time < cutoff_time].index))
f_Dz = np.asarray(Data['msd_z'].drop(Data[Data.Time < cutoff_time].index))
f_D = np.asarray(Data['msd'].drop(Data[Data.Time < cutoff_time].index))

# print(t)
# print(f_Dx)

model_Dx = LinearRegression().fit(t,f_Dx)
model_Dy = LinearRegression().fit(t,f_Dy)
model_Dz = LinearRegression().fit(t,f_Dz)
model_D = LinearRegression().fit(t,f_D)

Self_Diff_Data = pd.DataFrame(index=['r_sq','intercept','slope','diffusivity'],columns=['x','y','z','full'])
Self_Diff_Data['x'] = [model_Dx.score(t,f_Dx),model_Dx.intercept_,model_Dx.coef_[0],model_Dx.coef_[0] / 2]
Self_Diff_Data['y'] = [model_Dy.score(t,f_Dy),model_Dy.intercept_,model_Dy.coef_[0],model_Dy.coef_[0] / 2]
Self_Diff_Data['z'] = [model_Dz.score(t,f_Dz),model_Dz.intercept_,model_Dz.coef_[0],model_Dz.coef_[0] / 2]
Self_Diff_Data['full'] = [model_D.score(t,f_D),model_D.intercept_,model_D.coef_[0],model_D.coef_[0] / 2]
print(Self_Diff_Data)

Self_Diff_Data.to_csv('diff.dat')
print('Diffusivity calculation data written to: diff.dat')