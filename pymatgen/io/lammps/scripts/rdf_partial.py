#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 20:12:03 2019

@author: maximmakeev
"""

import os
import sys
import math
#
#
#-------------------------------------------------------------------
ConConstant = 1.660538921
NumConstant = float( 4.0/3.0 )
def main():
#---Partial RDFs are computed between types: nrdf[n] and nrdf[m]----
# 
    nrdf0 = []
    nrdf1 = []
    mass  = []
#-------------------------------------------------------------------
#Containers for atomic ID, type and Cartesian coordinates
#-------------------------------------------------------------------
    currwd = os.getcwd( )
    print("Current directory is:\n", currwd)
    rdfinput = open( "input.dat","r" )
#
    AI = rdfinput.readlines()
#
    rdfinput.close()
#----Parameters-loaded from the input file-------------------------
#
    tmp01 = AI[1].split()
    rcut = float( tmp01[0] )
    tmp02 = AI[3].split()
    ddr = float( tmp02[0] )
    tmp03 = AI[5].split()
    NA = int ( tmp03[0] )
    tmp04 = AI[7].split()
    ntypes = int ( tmp04[0] )
    tmp05 = AI[10].split()
    for kk in range( 0,ntypes ):
        mass.append( tmp05[kk] )
    tmp06 = AI[12].split()
    nrdfs = int( tmp06[0] )
    tmp07 = AI[15].split()
    tmp08 = AI[16].split()
    for ll in range( 0,nrdfs ):
        nrdf0.append( tmp07[ll] )
        nrdf1.append( tmp08[ll] )
    tmp09 = AI[19].split()
    input_mode = tmp09[0]
    filename   = tmp09[1]
    tmp10 = AI[21].split()
    workdir0 = tmp10[0]
#
    os.chdir( workdir0 )
    pathdir = os.getcwd()
    print("Working directory is:")
    print( pathdir )
#
    nbin = int(rcut/ddr) + 1
#
#----Find the MD trajectory files to be used for calculation-------
#
#---For configurations in multiple files, count the number of files
    if (input_mode == 'multi' ):
        listFiles = [ ]
        listoffiles = os.listdir()
        for entry in listoffiles:
            if entry.startswith( filename ):
                listFiles.append(entry)
                num_files = len( listFiles )
        print("# of files/frames (num_files) read:", num_files )
#
#---For configurations in a single file, count the number of frames
#
    num_frame_tot = int( 0 )
    if ( input_mode == 'single' ):
        filenameONE = filename
        rdffile = open( filenameONE,"r" )
        B = rdffile.readlines()
        rdffile.close()
        file_size = len( B )
        print( file_size )
        for ii in range( 0,file_size ):
            tmp = B[ii].split()
            if ( len( tmp ) > 1 and tmp[1] == "TIMESTEP" ): num_frame_tot += 1
            n_per_frame = int( file_size/num_frame_tot )
            num_files = int( num_frame_tot )
        print( "num_files/frames (num_files) read:=:", num_files )
#-----------------------------------------------------------------
#
#--Define containers to be used to sum configurations------------- 
    RDF_FULL_SUM = [ ]
    RDF_P_SUM    = [ ]
#
    for i in range( 0,nbin ):
        RDF_FULL_SUM.append( float(0.0) )
#    
    for i in range(0,len(nrdf0) ):
        RDF_P_SUM.append( [0.0]*nbin )
#
#----Loop over the coordinate files-------------------------------
#----Loop over trajectory files-----------------------------------
    for i_tr in range( 0,int( num_files ) ):
        if (input_mode == 'multi' ):
            filenameT = listFiles[ i_tr ]
            print( "Processing file:", filenameT )
            rdffile = open( filenameT,"r" )
            A = rdffile.readlines()   
            rdffile.close()
        if (input_mode == 'single' ):
            filenameT = filenameONE
            A = []
            print( "Processing frame:", i_tr )
            start_line = int(n_per_frame)*(i_tr)
            end_line   = int(n_per_frame)*int(i_tr + int(1) ) 
            for jjj in range( start_line, end_line ):
                A.append( B[ jjj ] ) 
        nlines = len( A )
        print( nlines)
        lco = [0]*5
        natom = 0
        key1 = 'ATOMS'
        ATID = []
        ATTY = []
        ATXX = []
        ATYY = []
        ATZZ = []
        for i in range( 0,nlines ):
            tmp = A[i].split()
            if len(tmp) > 1 and tmp[1] == key1:
                m = len( tmp )
                for n in range( 0,m ):
                    if tmp[n] == 'id':
                        lco[0] = (n-2)
                    if tmp[n] == 'type':
                        lco[1] = (n-2)
                    if tmp[n] == 'x':
                        lco[2] = (n-2)
                    if tmp[n] == 'y':
                        lco[3] = (n-2)
                    if tmp[n] == 'z':
                        lco[4] = (n-2)
            if( i == 3): NA = int( tmp[0] )
            if( i == 3): print( 'Number of atoms=: {0:d}'.format(NA) )
            if( i == 5): Lx = float( tmp[1] ) - float( tmp[0] )
            if( i == 6): Ly = float( tmp[1] ) - float( tmp[0] )
            if( i == 7): Lz = float( tmp[1] ) - float( tmp[0] )
            if( i >= 9 ):
                natom += 1
                ATID.append( int(  tmp[lco[0]] ) )
                ATTY.append( int(  tmp[lco[1]] ) )
                ATXX.append( float(tmp[lco[2]] ) )
                ATYY.append( float(tmp[lco[3]] ) )
                ATZZ.append( float(tmp[lco[4]] ) )
#
        if( NA != natom):
            print( "Major consistency check failed:" )
            print( "Configuration was not read correctly." )
            print( "NA=:", NA, "natoms=:", natom)
            sys.exit()
        n_a_pairs = len( nrdf0 )
        setID = {*()}
        atomtypes = [0]*( ntypes+1 )
        rho_n_pairs = [0]*n_a_pairs
#
        for jj in range( 0,natom ):
            npp = ATTY[jj]
            setID.add( npp )
            atomtypes[npp] += 1
            nset = len( setID )
#----------------------------------------------------------------------------
        if( ntypes != nset):
            print( "Consistency check failed:" )
            print( "Number of atomic types in the config file is \
                  different from the corresponding value in input file" )
            print( "ntypes=:", ntypes, "nset=:", nset)
#------------------------------------------------------------------------           
#---Start-Average Density Calculations-----------------------------------
#
        massT = float( 0.0 )
        for ii in range( 0,ntypes ):
            massT = massT + float( mass[ii] )*float( atomtypes[ ii+1 ] ) 
#
        volume = (Lx*Ly*Lz)
        densT = float( (massT/volume)*ConConstant )
        print( '{0:s}{1:10.8f}'.format('Average density=:', float(densT) )) 
#------------------------------------------------------------------------   
#---End--Average Density Calculations------------------------------------             
        RDF_COOR = []
        RDF_FULL = []
        RDF_P    = []
#    
        for i in range(0,len(nrdf0) ):
            RDF_P.append( [0.0]*nbin )
# 
#------Start radial distribution function calculations-------------------
        rho = natom/(Lx*Ly*Lz)
#
        for kk in range( 0, n_a_pairs ):
            ncurr = int( nrdf1[kk] ) 
            rho_n_pairs[ kk ] = float( atomtypes[ ncurr ] )/( Lx*Ly*Lz )
#       print( kk, nrdf0[kk], rho_n_pairs[ kk ] )
        if rho_n_pairs[kk] < float(1.0e-22):
            print("Error: Density is zero at kk=:", kk)
            sys.exit()
#------------------------------------------------------------------------
        for i in range( 0, nbin ):
            rr = float( ddr*(float(i) + float(0.5))  )
            RDF_COOR.append( rr )
            RDF_FULL.append( float(0.0) )
        for i in range(0, natom-1):
            if( i%1000 == 0): 
                print("Processing atom #:", i )
            for j in range(i+1, natom):
                dx = (ATXX[i] - ATXX[j])
                dy = (ATYY[i] - ATYY[j])
                dz = (ATZZ[i] - ATZZ[j])
                if( dx >  float(0.5)*Lx ): dx = dx - Lx
                if( dy >  float(0.5)*Ly ): dy = dy - Ly
                if( dz >  float(0.5)*Lz ): dz = dz - Lz
                if( dx < -float(0.5)*Lx ): dx = dx + Lx
                if( dy < -float(0.5)*Ly ): dy = dy + Ly
                if( dz < -float(0.5)*Lz ): dz = dz + Lz
                rsq = math.sqrt( dx*dx + dy*dy + dz*dz )
                if( rsq < rcut):
                    bin_num = int( rsq/ddr )
                    RDF_FULL[bin_num] += 2
                    for kl in range( 0,n_a_pairs ):
                        nta1 = int( nrdf0[kl] ) 
                        nta2 = int( nrdf1[kl] )
                        if( int(ATTY[i]) == nta1 and int(ATTY[j]) == nta2 ):
                            RDF_P[kl][bin_num] += 1
                        if( int(ATTY[j]) == nta1 and int(ATTY[i]) == nta2 ):
                            RDF_P[kl][bin_num] += 1
#          
        print( "Done computing RDF #:Mg_2TFSI_G1.lammpstrj", i_tr )
#---Normalization Procedure for the full RDF and partical RDFs
#
        for k in range( 0, nbin ):
            const = \
            (NumConstant*math.pi)*pow(ddr,3)*(pow((k+1),3) - pow(k,3) )*rho
            RDF_FULL[k] = RDF_FULL[k]/( const*natom ) 
#
#
        for kl in range( 0, n_a_pairs ):
            npp = int( nrdf0[ kl ] )
            for k in range( 0, nbin ):
                const = NumConstant*math.pi*pow(ddr,3)*( pow((k+1),3) \
                    - pow(k,3) )*rho_n_pairs[kl]
                RDF_P[kl][k] = RDF_P[kl][k]/(const*atomtypes[ npp ] )                     
#    
#----------------------------------------------------------------------
        for i in range(0, nbin):
            RDF_FULL_SUM[i] =  RDF_FULL_SUM[i] + RDF_FULL[i]
            for kl in range( 0,n_a_pairs ):
                RDF_P_SUM[kl][i] =  RDF_P_SUM[kl][i] + RDF_P[kl][i]   
#
        for i in range(0, nbin):
                RDF_FULL_SUM[i] =  RDF_FULL_SUM[i]/float( num_files  )
                for kl in range( 0,n_a_pairs ):
                    RDF_P_SUM[kl][i] =  RDF_P_SUM[kl][i]/float( num_files  )
#
#---------------------------------------------------------------------
#
    outf1 = open("rdf_full.dat","w+")
    for i in range(0,(nbin-1)):
        outf1.write("%25.20f, %25.20f\n" % (float(RDF_COOR[i]),\
                                            float(RDF_FULL_SUM[i])) )
    outf1.close()
    print("DONE")
    for kk in range( 0, n_a_pairs ):
        p1 = nrdf0[ kk ]
        p2 = nrdf1[ kk ]
        filename_p = "rdf_"+str(p1)+str(p2)+".dat"
        outf = open( filename_p, "w+" )
        for i in range( 0,nbin ):
            outf.write("%25.20f, %25.20f\n" % (float(RDF_COOR[i]),\
                                                  float(RDF_P_SUM[kk][i])) )
        outf.close()
#---------------------------------------------------------------------
#
    print("Full RDF and partial RDFs are written to RDF_NM.dat files")     
  
main()   
    
    
    
    
    
    
    
    
    
    
    
    

