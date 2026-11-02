import numpy as np
import sunpy as sp
import sunpy.map as sunmap
# import sunpy.data.sample
import sunpy.map
from sunpy.physics.differential_rotation import diff_rot, solar_rotate_coordinate


from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import (skycoord_to_pixel, pixel_to_skycoord,
                                _has_distortion, wcs_to_celestial_frame)
from astropy.coordinates import SkyCoord
import astropy.units as u

import os
import glob
import sys
import pandas as pd
import math
import cv2
from random import randint, randrange

from scipy import ndimage
from scipy.ndimage.interpolation import rotate

import skimage
import skimage.io as io
from skimage.measure import label, regionprops

import sqlalchemy

import math

import matplotlib.animation as animation
from matplotlib import rc
import matplotlib.patches as patches 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pymysql.cursors

from rapidfuzz.string_metric import levenshtein
from bisect import bisect

import ipywidgets as widgets

import tqdm

import matplotlib.lines as mlines

import warnings




def datetime_to_drawing_name(datetime):
    return f'usd{datetime["year"]}{datetime["month"]}{datetime["day"]}{datetime["hours"]}{datetime["minutes"]}-HO-AV.jpg'

def drawing_name_to_datetime(name):
    name2= name
    name2 = name2.replace('usd','')
    name2 = name2.split('-')[0]

    year = name2[0:4]
    month = name2[4:6]
    day = name2[6:8]
    hours = name2[8:10]
    minutes = name2[10:12]

    return {'year': year, 'month': month, 'day':day, 'hours':hours, 'minutes':minutes}

def datetime_to_whiteLight_string(datetime):
    name = 'UPH'
    name += f'{datetime["year"]}{datetime["month"]}{datetime["day"]}{datetime["hours"]}{datetime["minutes"]}'
    name += '.FTS'

    return name


def whiteLight_string_to_datetime(name):
    name2= name
    name2 = name2.replace('UPH','')
    year = name2[0:4]
    month = name2[4:6]
    day = name2[6:8]
    hours = name2[8:10]
    minutes = name2[10:12]
    return {'year': year, 'month': month, 'day':day, 'hours': hours, 'minutes': minutes}

def find_closest_drawing(name, drawings_list):
    idx = bisect(drawings_list, name)
    return drawings_list[idx-1]

def find_closest_wl(name, whitelight_list):
    idx = bisect(whitelight_list, name)
    return whitelight_list[idx]

def datetime_to_db_string(datetime):
    date = '-'.join([datetime['year'],datetime['month'], datetime['day']])
    time = ':'.join([datetime['hours'], datetime['minutes'], '00'])
    
    return f'{date} {time}'

def query_table(datetime_str, database, table):
    connection = pymysql.connect(host=host,
                            user=user,
                            password=password,
                            database=database,
                            cursorclass=pymysql.cursors.DictCursor)
    with connection:
        with connection.cursor() as cursor:
            sql = f'SELECT * FROM {database}.{table} WHERE DateTime="{datetime_str}";'
            # print(sql)
            cursor.execute(sql)
            result = cursor.fetchall()
            # print(result.keys())
            return result

from sunpy.coordinates import frames , transformations
def coordinates2pixel(fits_file, Longitude, Latitude):
    print(fits_file)
    hdulst = fits.open(fits_file)[0]
    hdulst.header.append(('CTYPE1', 'HPLN-TAN'))
    hdulst.header.append(('CTYPE2', 'HPLT-TAN'))
    
    uset_map = sunmap.GenericMap(hdulst.data, hdulst.header)

    hgc_coord = SkyCoord(Longitude*u.rad, Latitude*u.rad,
                        obstime=uset_map.date.value,
                        frame=frames.HeliographicCarrington,
                        observer="earth")
    
    hp_coord = hgc_coord.transform_to(uset_map.coordinate_frame)
    
    wcs = WCS(hdulst.header)
    pixels = skycoord_to_pixel(hgc_coord,wcs,0)
    pixels = np.array(list(pixels))

    
    return pixels

def show_sunspots(mask,props):
    plt.figure()
    plt.imshow(mask)
    for prop in props:
        plt.scatter(prop.centroid[1],prop.centroid[0], c='r')

def get_sunspots(mask):
    m2 = label(mask)
    props_spots = regionprops(m2)
    
    centers = []
    for prop in props_spots:
        center = np.array(prop.centroid)
        centers.append(center)
    return centers

def propagate_known_IDs(prev, current):
    centerIDs = [{"center": item, 'ID': None} for item in current]
    
    if len(prev) == 0: # usually, only when processing first frame in sequence
        return centerIDs
        
    prev_centers_arr = np.array([it['center'] for it in prev])
   
    # if item in current is closer than a distance threshold give same id, else leave ID=None
    for sunspot in centerIDs:
        
        distances = np.linalg.norm(np.array(sunspot["center"]) - prev_centers_arr, axis=1)
        
        closest_prev = np.min(distances)
        closest_prev_idx = np.argmin(distances)
        print(f'{sunspot["center"]} -> closest is {prev[closest_prev_idx]["ID"]} at {prev[closest_prev_idx]["center"]}')
    
        ## Shouldn't the threshold be time/sun revolution dependent? (time between samples changes the 'valid' range)
        thresh = 25
        if closest_prev < thresh:
            sunspot['ID'] = prev[closest_prev_idx]['ID']
    return centerIDs


def generate_IDs_to_unknowns(centerIDs, used_IDS):
    IDs = [item['ID'] for item in centerIDs]
    
    for item in centerIDs:
        if item['ID'] is None:
            # generate newID
            tmp = None
            while (tmp is None) or (tmp in IDs) or (tmp in used_IDS):
                tmp = randrange(1000,2000) #maximum endpoint is exclusive -> never 2000
            item["ID"] = tmp
            used_IDS.append(tmp)
            
    return centerIDs


def rotate_pt(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy



def diff_rot_next_pos(umap1, umap2, wcs, coords):
    future_date = umap2.date
    
    rot_coords = []
    for this_hpc_x, this_hpc_y in coords:
        
        start_coord = pixel_to_skycoord(this_hpc_y, this_hpc_x, wcs, origin=0)
        start_coord = SkyCoord(start_coord.Tx, start_coord.Ty,
                               frame=umap1.coordinate_frame)
        rotated_coord = solar_rotate_coordinate(start_coord, time=future_date)
        coord = SkyCoord([start_coord.Tx, rotated_coord.Tx],
                         [start_coord.Ty, rotated_coord.Ty],
                         frame=umap1.coordinate_frame)
        rot_coords.append(coord)
    return rot_coords

# +
def open_and_add_celestial(filename):
    hdulst1 = fits.open(filename)[0]
    hdulst1.header.append(('CTYPE1', 'HPLN-TAN'))
    hdulst1.header.append(('CTYPE2', 'HPLT-TAN'))
    
    uset_map1 = sunmap.GenericMap(hdulst1.data, hdulst1.header)
    return uset_map1, hdulst1.header

def open_and_add_celestial2(filename, date_obs=None):
    hdulst1 = fits.open(filename)[0]
    hdulst1.header.append(('CTYPE1', 'HPLN-TAN'))
    hdulst1.header.append(('CTYPE2', 'HPLT-TAN'))
    if date_obs is not None:
        hdulst1.header.append(('DATE-OBS', date_obs))
        
        hdulst1.header.set('OBS_MODE', 'bi q2')
        
    uset_map1 = sunmap.GenericMap(hdulst1.data, hdulst1.header)
    return uset_map1, hdulst1.header


# -

def show_step(uset_map1, uset_map2, est_coords):
    fig = plt.figure(figsize=(6,3))
    ax = plt.subplot(121,projection=uset_map1)
    uset_map1.plot(clip_interval=(1, 99.99)*u.percent)
    uset_map1.draw_grid()

    for coord in est_coords:
        ax.plot_coord(coord, 'o-', alpha=.5)
    plt.ylim(0, uset_map1.data.shape[1])
    plt.xlim(0, uset_map1.data.shape[0])

    ax2 = plt.subplot(122,projection=uset_map2)
    uset_map2.plot(clip_interval=(1, 99.99)*u.percent)
    uset_map2.draw_grid()

    for coord in est_coords:
        ax2.plot_coord(coord, 'o-', alpha=.5)
    plt.ylim(0, uset_map2.data.shape[1])
    plt.xlim(0, uset_map2.data.shape[0])

    return

def skycoord2pixels(header, coords):
    wcs = WCS(header)
    
    pixels_coords = []
    for coord in coords:
        pixels = skycoord_to_pixel(coord,wcs,0)
        pixels = np.array(list(pixels))
        pixels_coords.append(pixels)
    
    return pixels_coords

def propagate_known_IDs_2(prev, current):
    centerIDs = [{"center": item, 'ID': None} for item in current]
    
    if len(prev) == 0: # usually, only when processing first frame in sequence
        return centerIDs

    prev_centers_arr = np.array([ np.array([it['center'][0], it['center'][1]]) for it in prev])
    
    # if item in current is closer than a distance threshold give same id, else leave ID=None
    for sunspot in centerIDs:
        distances = np.linalg.norm(np.array(sunspot["center"]) - prev_centers_arr, axis=1)
        closest_prev = np.min(distances)
        closest_prev_idx = np.argmin(distances)        
    
        
        ids = [s['ID'] for s in centerIDs]
        
        thresh = 25
        if (closest_prev < thresh) and (prev[closest_prev_idx]['ID'] not in ids):
            sunspot['ID'] = prev[closest_prev_idx]['ID']

    return centerIDs

