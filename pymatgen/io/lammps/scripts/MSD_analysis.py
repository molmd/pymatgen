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
parser.add_argument('--msd_type','-mt',help='type of method used to calculate MSDs. Can be com (center of mass) or allatom. Defaults to com')

args = parser.parse_args()

if args.file_pattern:
    file_pattern = args.file_pattern + '*'
else:
    file_pattern = '*.lammpstrj'

if args.timestep:
    fs_per_step = float(args.timestep)
else:
    fs_per_step = 1

if args.cutoff_time:
    cutoff_time = int(args.cutoff_time)
else:
    cutoff_time = 500

if args.msd_type:
    msd_type = args.msd_type
else:
    msd_type = 'com'

'''Values used for testing'''
# file_pattern = 'trj_files/msd_files/dump.0.5_dhps_2.5_naoh_spce.nvt_production.*.lammpstrj'
# fs_per_step = 0.5
# msd_type = 'allatom'

print('File pattern used: ' + file_pattern)
print('Timestep used: ' + str(fs_per_step) + ' fs')
print('Cutoff time used: ' + str(cutoff_time) + ' ps')
print('MSD type used: ' + msd_type)
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
    if msd_type == 'allatom':
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
    elif msd_type == 'com':
        # # Creating initial position information of CoM of molecules in DataFrame then adding msd data to Data object
        if i == 0:
            init_n_mols = dump.data['mol'].max()
            print('Number of molecules in system: ' + str(init_n_mols))
            Initial_data = pd.concat([pd.DataFrame([i+1],columns=['Mol']) for i in range(init_n_mols)],ignore_index=True)

            for mol in range(init_n_mols):
                init_dump_data = dump.data[dump.data['mol'] == mol + 1]
                x_com = init_dump_data['xu'].multiply(init_dump_data['mass']).sum() / init_dump_data['mass'].sum()
                y_com = init_dump_data['yu'].multiply(init_dump_data['mass']).sum() / init_dump_data['mass'].sum()
                z_com = init_dump_data['zu'].multiply(init_dump_data['mass']).sum() / init_dump_data['mass'].sum()
                Initial_data.at[mol, 'x_com'] = x_com
                Initial_data.at[mol, 'y_com'] = y_com
                Initial_data.at[mol, 'z_com'] = z_com

            Initial_data[['dx2','dy2','dz2']] = Initial_data[['x_com','y_com','z_com']].subtract(Initial_data[['x_com','y_com','z_com']],axis=1).pow(2)
            Initial_data['disp2'] = Initial_data[['dx2','dy2','dz2']].sum(axis=1)

            time = dump.timestep * ps_per_step
            msd_x = Initial_data['dx2'].mean()
            msd_y = Initial_data['dy2'].mean()
            msd_z = Initial_data['dz2'].mean()
            msd = Initial_data['disp2'].mean()

            Data = Data.append({'Time':time,'msd_x':msd_x,'msd_y':msd_y,'msd_z':msd_z,'msd':msd},ignore_index=True)

        # # Creating position information of CoM of molecules for timesteps later than the first one in DataFrame then adding MSD data to Data object
        else:
            n_mols = dump.data['mol'].max()
            assert(n_mols == init_n_mols), 'Different frames have different numbers of molecules.'
            Current_data = pd.concat([pd.DataFrame([i+1],columns=['Mol']) for i in range(n_mols)],ignore_index=True)

            for mol in range(n_mols):
                current_dump_data = dump.data[dump.data['mol'] == mol + 1]
                x_com = current_dump_data['xu'].multiply(current_dump_data['mass']).sum() / current_dump_data['mass'].sum()
                y_com = current_dump_data['yu'].multiply(current_dump_data['mass']).sum() / current_dump_data['mass'].sum()
                z_com = current_dump_data['zu'].multiply(current_dump_data['mass']).sum() / current_dump_data['mass'].sum()
                Current_data.at[mol, 'x_com'] = x_com
                Current_data.at[mol, 'y_com'] = y_com
                Current_data.at[mol, 'z_com'] = z_com

            Current_data[['dx2','dy2','dz2']] = Current_data[['x_com','y_com','z_com']].subtract(Initial_data[['x_com','y_com','z_com']],axis=1).pow(2)
            Current_data['disp2'] = Current_data[['dx2','dy2','dz2']].sum(axis=1)

            time = dump.timestep * ps_per_step
            msd_x = Current_data['dx2'].mean()
            msd_y = Current_data['dy2'].mean()
            msd_z = Current_data['dz2'].mean()
            msd = Current_data['disp2'].mean()

            Data = Data.append({'Time':time,'msd_x':msd_x,'msd_y':msd_y,'msd_z':msd_z,'msd':msd},ignore_index=True)


# print(Data.head(11))

# # Write to csv file
out_file_name = 'msd_' + msd_type + '.dat'
Data.to_csv(out_file_name,index=False)
print('MSD data written to: ' + out_file_name)

# Test_data = pd.read_csv('msd.dat')
# print(Test_data.head(11))

# # Linear regression
t = np.asarray(Data['Time'].drop(Data[pd.to_numeric(Data.Time) < cutoff_time].index)).reshape((-1,1))
f_Dx = np.asarray(Data['msd_x'].drop(Data[pd.to_numeric(Data.Time) < cutoff_time].index))
f_Dy = np.asarray(Data['msd_y'].drop(Data[pd.to_numeric(Data.Time) < cutoff_time].index))
f_Dz = np.asarray(Data['msd_z'].drop(Data[pd.to_numeric(Data.Time) < cutoff_time].index))
f_D = np.asarray(Data['msd'].drop(Data[pd.to_numeric(Data.Time) < cutoff_time].index))

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
Self_Diff_Data['full'] = [model_D.score(t,f_D),model_D.intercept_,model_D.coef_[0],model_D.coef_[0] / 6]
print(Self_Diff_Data)

diff_file_name = 'diff_' + msd_type + '.dat'
Self_Diff_Data.to_csv(diff_file_name)
print('Diffusivity calculation data written to: ' + diff_file_name)