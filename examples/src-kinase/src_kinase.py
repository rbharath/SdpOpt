"""Src Kinase Dataset
"""
from __future__ import print_function, absolute_import, division

from glob import glob
from io import BytesIO
from os import makedirs
from os.path import exists
from os.path import join
from zipfile import ZipFile
import numpy as np
import ntpath
try:
    # Python 2
    from urllib2 import urlopen
except ImportError:
    # Python 3+
    from urllib.request import urlopen

import mdtraj as md

# Add this back in once uploaded to figshare
#DATA_URL = "http://downloads.figshare.com/article/public/1026131"
TARGET_DIRECTORY = "/home/rbharath/mixtape_data/src_kinase/Trajectories/"
SOURCE_DIRECTORY = "/home/shukla/Src_Kinase_Data_Pande_Group"

def src_kinase_atom_indices():
    """
    K 36 -- atoms 562 - 583 -> (22 atoms)
    E 51 -- atoms 797 - 811 -> (15 atoms)
    D 145 -- start atom 2318
    F 146
    G 147 -- end atom 2356 -> (39 atoms)
    R 150 -- atoms 2386 - 2409 -> (24 atoms)
    Y 157 -- atoms 2504 - 2524  -> (21 atoms)
    Total: 121 atoms
    """
    K_36_atoms = 22
    E_51_atoms = 15
    dfg_145_147_atoms = 39
    R_150_atoms = 24
    Y_157_atoms = 21
    _atoms = (K_36_atoms + E_51_atoms + dfg_145_147_atoms 
                + R_150_atoms + Y_157_atoms)
    indices = [] 
    pos = 0
    indices += range(562, 583+1)
    indices += range(797, 811+1)
    indices += range(2318, 2356+1)
    indices += range(2386, 2409+1)
    indices += range(2504, 2524+1)
    return indices

def fetch_src_kinase(data_home=None):
    """Loader for the src kinase dataset

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all mixtape data is stored in '~/mixtape_data' subfolders.

    download_if_missing: optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Notes
    -----
    This dataset contains 10 MD trajectories
    """
    data_dir = SOURCE_DIRECTORY 
    print("data_dir: %s"%data_dir)
    save_dir = TARGET_DIRECTORY
    print("save_dir: %s"%save_dir)
    atom_indices = src_kinase_atom_indices()
    locs = glob(join(data_dir, 'Trajectories/trj*.lh5'))
    top = md.load(join(data_dir, 'protein_8041.pdb'))

    trajectories = []
    count = 0
    delta=10
    for fn in locs:
        print("File %d: %s" % (count, fn))
        traj = md.load(fn, top=top)
        traj.restrict_atoms(atom_indices)
        traj = traj[::delta]
        fname = ntpath.basename(fn)
        save_loc = join(save_dir, fname)
        traj.save(save_loc)
        count += 1
    return
