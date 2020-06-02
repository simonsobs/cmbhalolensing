"""
Subtracting model clusters from the map, using nemo output catalog.

"""
import os
import sys
import IPython
import nemoCython
import astropy.io.fits as pyfits
import astropy.table as atpy
import numpy as np
from astLib import *
from pixell import enmap
from nemo import maps, catalogs, signals


def temp(shape, wcs, catalog, beamFileName, obsFreqGHz = None, GNFWParams = 'default', 
                   cosmoModel = None, applyPixelWindow = True, override = None):
    """Make a map with the given dimensions (shape) and WCS, containing model clusters or point sources, 
    with properties as listed in the catalog. This can be used to either inject or subtract sources
    from real maps.
    
    Args:
        shape (tuple): The dimensions of the output map (height, width) that will contain the model sources.
        wcs (:obj:`astWCS.WCS`): A WCS object that defines the coordinate system of the map. 
        catalog (:obj:`astropy.table.Table`): An astropy Table object containing the catalog. This must 
            include columns named 'RADeg', 'decDeg' that give object coordinates. For point sources, the 
            amplitude in uK must be given in a column named 'deltaT_c'. For clusters, either 'M500' (in 
            units of 10^14 MSun), 'z', and 'fixed_y_c' must be given (as in a mock catalog), OR the 
            catalog must contain a 'template' column, with templates named like, e.g., Arnaud_M1e14_z0p2
            (for a z = 0.2, M500 = 1e14 MSun cluster; see the example .yml config files included with nemo).
        beamFileName: Path to a text file that describes the beam.
        obsFreqGHz (float, optional): Used only by cluster catalogs - if given, the returned map will be 
            converted into delta T uK, assuming the given frequency. Otherwise, a y0 map is returned.
        GNFWParams (str or dict, optional): Used only by cluster catalogs. If 'default', the Arnaud et al. 
            (2010) Universal Pressure Profile is assumed. Otherwise, a dictionary that specifies the profile
            parameters can be given here (see gnfw.py).
        override (dict, optional): Used only by cluster catalogs. If a dictionary containing keys
            {'M500', 'redshift'} is given, all objects in the model image are forced to have the 
            corresponding angular size. Used by :meth:`positionRecoveryTest`.
        applyPixelWindow (bool, optional): If True, apply the pixel window function to the map.
            
    Returns:
        Map containing injected sources.
    
    """
    
    modelMap=np.zeros(shape, dtype = float)
    
    # This works per-tile, so throw out objects that aren't in it
    catalog=catalogs.getCatalogWithinImage(catalog, shape, wcs)

    if cosmoModel is None:
        cosmoModel=signals.FlatLambdaCDM(H0 = 70.0, Om0 = 0.3, Ob0 = 0.05, Tcmb0 = signals.TCMB)
    
    # Set initial max size in degrees from beam file (used for sources; clusters adjusted for each object)
    numFWHM=5.0
    beam=signals.BeamProfile(beamFileName = beamFileName)
    maxSizeDeg=beam.rDeg[np.argmin(abs(beam.profile1d-0.5))]*2*numFWHM 
    
    # Map of distance(s) from objects - this will get updated in place (fast)
    degreesMap=np.ones(modelMap.shape, dtype = float)*1e6
    
    if 'fixed_y_c' in catalog.keys():
        # Clusters: for speed - assume all objects are the same shape
        if override is not None:
            fluxScaleMap=np.zeros(modelMap.shape)
            for row in catalog:
                degreesMap, xBounds, yBounds=nemoCython.makeDegreesDistanceMap(degreesMap, wcs, 
                                                                            row['RADeg'], row['decDeg'], 
                                                                            maxSizeDeg)
                fluxScaleMap[yBounds[0]:yBounds[1], xBounds[0]:xBounds[1]]=row['fixed_y_c']*1e-4
            theta500Arcmin=signals.calcTheta500Arcmin(override['redshift'], override['M500'], cosmoModel)
            maxSizeDeg=5*(theta500Arcmin/60)
            modelMap=signals.makeArnaudModelSignalMap(override['redshift'], override['M500'], degreesMap, 
                                                      wcs, beam, GNFWParams = GNFWParams,
                                                      maxSizeDeg = maxSizeDeg, convolveWithBeam = False)
            modelMap=modelMap*fluxScaleMap
            modelMap=maps.convolveMapWithBeam(modelMap, wcs, beam, maxDistDegrees = 1.0)

        # Clusters - insert one at a time (with different scales etc.) - currently taking ~1.6 sec per object
        else:
            count=0
            for row in catalog:
                count=count+1
                print("... %d/%d ...abc" % (count, len(catalog)))
                # NOTE: We need to think about this a bit more, for when we're not working at fixed filter scale
                if 'true_M500' in catalog.keys():
                    M500=row['true_M500']*1e14
                    z=row['redshift']
                    y0ToInsert=row['fixed_y_c']*1e-4
                else:
                    if 'template' not in catalog.keys():
                        raise Exception("No M500, z, or template column found in catalog.")
                    bits=row['template'].split("#")[0].split("_")
                    M500=float(bits[1][1:].replace("p", "."))
                    z=float(bits[2][1:].replace("p", "."))
                    y0ToInsert=row['y_c']*1e-4  # or fixed_y_c...
                theta500Arcmin=signals.calcTheta500Arcmin(z, M500, cosmoModel)
                maxSizeDeg=5*(theta500Arcmin/60)
                degreesMap=np.ones(modelMap.shape, dtype = float)*1e6

                if row['SNR'] > 20:

                    for i in range(nname.size):
                        if nname[i] == row['name']:
                            new_ra = nra[i]
                            new_dec = ndec[i]
                            break

                    degreesMap, xBounds, yBounds=nemoCython.makeDegreesDistanceMap(degreesMap, wcs, 
                                                                        new_ra, new_dec, 
                                                                        maxSizeDeg)
                else:
                    degreesMap, xBounds, yBounds=nemoCython.makeDegreesDistanceMap(degreesMap, wcs, 
                                                                            row['RADeg'], row['decDeg'], 
                                                                            maxSizeDeg)


                modelMap=modelMap+signals.makeArnaudModelSignalMap(z, M500, degreesMap, wcs, beam, 
                                                                GNFWParams = GNFWParams, amplitude = y0ToInsert,
                                                                maxSizeDeg = maxSizeDeg, convolveWithBeam = False)
            modelMap=maps.convolveMapWithBeam(modelMap, wcs, beam, maxDistDegrees = 1.0)

    else:
        # Sources - note this is extremely fast, but will be wrong for sources close enough to blend
        fluxScaleMap=np.zeros(modelMap.shape)
        for row in catalog:
            degreesMap, xBounds, yBounds=nemoCython.makeDegreesDistanceMap(degreesMap, wcs, 
                                                                        row['RADeg'], row['decDeg'],    
                                                                    maxSizeDeg)
            fluxScaleMap[yBounds[0]:yBounds[1], xBounds[0]:xBounds[1]]=row['deltaT_c']
        modelMap=signals.makeBeamModelSignalMap(degreesMap, wcs, beam)
        modelMap=modelMap*fluxScaleMap

    # Optional: convert map to deltaT uK
    # This should only be used if working with clusters - source amplitudes are fed in as delta T uK already
    if obsFreqGHz is not None:
        modelMap=maps.convertToDeltaT(modelMap, obsFrequencyGHz = obsFreqGHz)
    
    # Optional: apply pixel window function - generally this should be True
    # (because the source-insertion routines in signals.py interpolate onto the grid rather than average)
    if applyPixelWindow == True:
        modelMap=enmap.apply_window(modelMap, pow = 1.0)

    return modelMap


# peak pixel coordinate for clusters of SNR > 20
altype = np.dtype([('f0','float'),('f1','float'),('f2','<U21')])
data0 = np.genfromtxt('data/coord_snr_20_150.txt', delimiter="\t", dtype=altype, usecols=(0,1,2))
#data0 = np.genfromtxt('data/coord_snr_20_90.txt', delimiter="\t", dtype=altype, usecols=(0,1,2))
nra = data0['f0']
ndec = data0['f1']
nname = data0['f2']

# beam file
beamFileName = "data/b20190809_s16_pa2_f150_nohwp_night_beam_tform_jitter_cmb.txt"
#beamFileName="data/b20190809_s16_pa3_f090_nohwp_night_beam_tform_jitter_cmb.txt"

# cluster catalogue
tab = atpy.Table().read("data/AdvACT_S18Clusters_v1.0-beta.fits")

# input map for tSZ subtraction
with pyfits.open("data/act_planck_s08_s18_cmb_f150_night_srcfree_map.fits") as img:

    d = img[0].data
    dd = d[0]
    wcs = astWCS.WCS(img[0].header, mode = 'pyfits')

#print(dd.shape)

m = temp(dd.shape, wcs, tab, beamFileName, obsFreqGHz = 150.0)

#astImages.saveFITS("model_ACT_Planck_night_150.fits", m, wcs)
astImages.saveFITS("modelSubtracted_ACT_Planck_night_150.fits", dd-m, wcs)
