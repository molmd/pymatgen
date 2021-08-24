import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MSD_data = pd.read_csv("msd.dat")

# print(MSD_data.head(11))

plt.figure(1)
plt.plot(MSD_data.Time, MSD_data.msd_x, c="red", label="x")
plt.plot(MSD_data.Time, MSD_data.msd_y, c="green", label="y")
plt.plot(MSD_data.Time, MSD_data.msd_z, c="blue", label="z")
plt.xlabel("t [ps]")
plt.ylabel("msd in 1 dim [Angst^2]")
plt.axis([0, MSD_data.Time.max(), 0, MSD_data.msd_x.max() * 1.2])
plt.title("MSDs in single dimensions")
plt.legend()
plt.savefig("1D_msds.png")

plt.figure(2)
plt.plot(MSD_data.Time, MSD_data.msd, c="red")
plt.xlabel("t [ps]")
plt.ylabel("msd in 1 dim [Angst^2]")
plt.axis([0, MSD_data.Time.max(), 0, MSD_data.msd.max() * 1.2])
plt.title("MSDs in 3D")
plt.savefig("3D_msd.png")
