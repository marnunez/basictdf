__doc__ = """
basictdf is a **read** and **write** parser for the BTS Bioengineering TDF file format. 
This format is tipically used as storage of raw data from a BTS motion capture acquisition system
(e.g. raw EMG, 2D raw marker data) but can also serve as storage for processed data
(e.g. 3D reconstructed marker data, filtered EMG signals, events).
"""
from basictdf.basictdf import Tdf
