import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data_114 = pd.read_csv('rdf_411.dat',header=None,names=['r [Ang]','g(r)'])

# print(Data_114)

"""Convert columns to np.ndarrays"""
# Radii = Data_114['r [Ang]'].values
# G_func = Data_114['g(r)'].values


def Parse_rdf_csv(filename):

    # Convert csv file to pandas.dataframe
    Data = pd.read_csv(filename, header=None, names=["r [Ang]", "g(r)"])
    # Convert columns in dataframe to numpy.ndarrays
    Radii = Data["r [Ang]"].values
    G = Data["g(r)"].values

    return Radii, G


# output = Parse_rdf_csv('rdf_1111.dat')
# print(output)
# print(type(output))
# print(len(output))


def Plot_rdf_data(Data_Tuple, fig=1, title_types="", Save=False, filename="rdf.dat"):

    plt.figure(fig)
    plt.plot(Data_Tuple[0], Data_Tuple[1], c="green")
    plt.xlabel("r [Ang]")
    plt.ylabel("g(r)")
    plt.axis([0, 5, 0, Data_Tuple[1].max() * 1.2])
    plt.title("rdf " + title_types)

    if Save:
        Save_filename = filename[:-3] + "png"
        plt.savefig(Save_filename)
    else:
        plt.show()


# Plot_rdf_data(output,title_types='Na - O (Wat)',Save=False,filename='rdf_119.dat')

# Plot_rdf_data(output,title_types='Na - Na',Save=True,filename='rdf_1111.dat')

Data_114 = Parse_rdf_csv("rdf_114.dat")
Plot_rdf_data(Data_114, fig=1, title_types="Na - O (DHPS o)", Save=True, filename="rdf_114.png")

Data_115 = Parse_rdf_csv("rdf_115.dat")
Plot_rdf_data(Data_115, fig=2, title_types="Na - O (DHPS oh)", Save=True, filename="rdf_115.png")

Data_117 = Parse_rdf_csv("rdf_117.dat")
Plot_rdf_data(Data_117, fig=3, title_types="Na - O (WAT)", Save=True, filename="rdf_117.png")

Data_119 = Parse_rdf_csv("rdf_119.dat")
Plot_rdf_data(Data_119, fig=4, title_types="Na - O (OH-)", Save=True, filename="rdf_119.png")

Data_1111 = Parse_rdf_csv("rdf_1111.dat")
Plot_rdf_data(Data_1111, fig=5, title_types="Na - Na", Save=True, filename="rdf_1111.png")

Data_full = Parse_rdf_csv("rdf_full.dat")
Plot_rdf_data(Data_full, fig=6, title_types="Full", Save=True, filename="rdf_full.png")
