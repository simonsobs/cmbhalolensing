import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import utils as cutils
from pixell import enmap, reproject, enplot, utils, wcsutils
from orphics import maps, mpi, io, stats,cosmology,lensing
from scipy.optimize import curve_fit
from numpy import save
import symlens
import healpy as hp
import os, sys
import time as t
from enlib import bench
import warnings
import re
#from szar import counts
#from HMFunc.cosmology import Cosmology

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Stacked CMB lensing.')
parser.add_argument("save_name", type=str,help='Root name for saving plots.')
parser.add_argument("stack_path", type=str,help='Stack relative path.')
parser.add_argument("mf_path", type=str,help='Meanfield relative path.')
parser.add_argument("--theory",     type=str,  default=None,help="Lensed theory location for comparison of sim.")
parser.add_argument("--cwidth",     type=float,  default=30.0,help="Crop width arcmin.")
parser.add_argument("--fwhm",     type=float,  default=3.5,help="FWHM for smoothing.")
parser.add_argument("--z",     type=float,  default=0.55,help="Redshift for profile fit.")
parser.add_argument("--conc",     type=float,  default=3.0,help="Concentration for profile fit.")
parser.add_argument("--sigma-mis",     type=float,  default=None,help="Miscentering Rayleigh width in arcmin.")
parser.add_argument("--mass-guess",     type=float,  default=2e14,help="Mass guess in solar masses.")
parser.add_argument("--snr-guess",     type=float,  default=10,help="SNR guess.")
parser.add_argument("--nsigma",     type=float,  default=10,help="Number of sigma away from mass-guess to evaulate likelihood at.")
parser.add_argument("--num-ms",     type=int,  default=100,help="Number of mass points to evaluate likelihood at.")
parser.add_argument("--arcmax",     type=float,  default=10.,help="Maximum arcminute radius distance for fit.")
parser.add_argument("--overdensity",     type=float,  default=200.,help="NFW mass definition overdensity.")
parser.add_argument("--critical", action='store_true',help='Whether NFW mass definition is wrt critical density (default: mean matter density).')
parser.add_argument("--at-z0", action='store_true',help='Whether NFW mass definition is at z=0 (default: at cluster redshift).')
parser.add_argument("--ymin",     type=float,  default=-0.02,help="Profile y axis scale minimum.")
parser.add_argument("--ymax",     type=float,  default=0.2,help="Profile y axis scale maximum.")
parser.add_argument("--plim",     type=float,  default=None,help="Stack plot limit.")
parser.add_argument("--slim",     type=float,  default=None,help="Stack plot limit (smoothed).")
parser.add_argument("--ignore-param", action='store_true',help='Ignore parameter matching errors.')


args = parser.parse_args()


mf_path = args.mf_path
stack_path  = args.stack_path
save_name = args.save_name
ignore_param = args.ignore_param
cwidth = args.cwidth

cutils.postprocess(stack_path,mf_path,save_name=save_name,ignore_param=ignore_param,args=args)
