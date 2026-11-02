# -*- coding: utf-8 -*-
import numpy as np
import sunpy as sp
import sunpy.map as sunmap

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle

from sunpy.physics.differential_rotation import diff_rot, solar_rotate_coordinate
from sunpy.coordinates import frames , transformations

from skimage.measure import label, regionprops, block_reduce

import os
import glob
import sys
import pandas as pd
import math
import cv2
import skimage

from scipy import ndimage 
import skimage.io as io
import sqlalchemy

import matplotlib.animation as animation
from matplotlib import rc
import matplotlib.patches as patches 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# from scipy.ndimage.interpolation import rotate

import pymysql.cursors

# from rapidfuzz.string_metric import levenshtein
from bisect import bisect

from datetime import datetime, timedelta

from tqdm.notebook import tqdm

import ipywidgets as widgets

import importlib
import tracking_utilities as utils #import the module here, so that it can be reloaded.
importlib.reload(utils)
import Class2Bbox as c2bb
importlib.reload(c2bb)

import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy

import sqlite3
import json

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

def whitelight_to_datetime(name):
    name2= name
    name2 = name2.replace('UPH','')

    year = name2[0:4]
    month = name2[4:6]
    day = name2[6:8]
    hours = name2[8:10]
    minutes = name2[10:12]
    seconds = name2[12:14]

    return {'year': year, 'month': month, 'day':day, 'hours':hours, 'minutes':minutes, "seconds": seconds}

def datetime_to_drawing_name(datetime):
    name = 'usd'
    name += f'{datetime["year"]}{datetime["month"]}{datetime["day"]}{datetime["hours"]}{datetime["minutes"]}'
    name += '-HO-AV.jpg'

    return name

def datetime_to_whiteLight_string(datetime):
    name = 'UPH'
    name += f'{datetime["year"]}{datetime["month"]}{datetime["day"]}{datetime["hours"]}{datetime["minutes"]}'
    name += '.FTS'
#     f"{number:02d}"

    return name

def datetime_to_db_string(datetime):
    date = '-'.join([datetime['year'],datetime['month'], datetime['day']])
    time = ':'.join([datetime['hours'], datetime['minutes'], '00'])
    
    return f'{date} {time}'


def db_string_to_datetime(db_string):
    date_str, time_str = db_string.split(" ") 
    year, month, day = date_str.split('-')
    hours, minutes, seconds = time_str.split(':')
    
    return {'year': year, 'month': month, 'day':day, 'hours':hours, 'minutes':minutes, "seconds": seconds}


def leftOrRightDrawing(t ,t_prev, t_next):
    best = None

    if    ( t_prev is None)   and (not t_next is None):
        return t_next
    elif (not t_prev is None) and   (t_next is None)  :
        return t_prev

    else:
        a = int(os.path.basename(t_prev)[3:-10])
        b = int(os.path.basename(t_next)[3:-10])

        c = int(os.path.basename(t)[3:-10])

#         print( 'prev: ', a, ', next: ', b, ', current: ' , c)

        d1 = np.abs(c-a)
        d2 = np.abs(c-b)

        if d1 <= d2:
            best = t_prev
        else:
            best = t_next

    return best

def find_closest_drawing(name, drawing_list):
    idx = bisect(drawing_list, name)

    a = None if idx < 1 else drawing_list[idx-1]
    b = None if idx >= len(drawing_list) else drawing_list[idx]
    
    return leftOrRightDrawing(name, a, b)

def leftOrRightWhitelight(t ,t_prev, t_next):
    best = None

    if    ( t_prev is None)   and (not t_next is None):
        return t_next
    elif (not t_prev is None) and   (t_next is None)  :
        return t_prev

    else:
#         print(t_prev,t_next,t)
        a = int(os.path.basename(t_prev)[3:-10])
        b = int(os.path.basename(t_next)[3:-10])

        c = int(os.path.basename(t)[3:-10])

#         print( 'prev: ', a, ', next: ', b, ', current: ' , c)

        d1 = np.abs(c-a)
        d2 = np.abs(c-b)

        if d1 <= d2:
            best = t_prev
        else:
            best = t_next

    return best

def find_closest_wl(name, wl_list):
    idx = bisect(wl_list, name)

    a = None if idx < 1 else wl_list[idx-1]
    b = None if idx >= len(wl_list) else wl_list[idx]
    
    return leftOrRightWhitelight(name, a, b)


def query_table(datetime_str, database, table):
    connection = pymysql.connect(host=host,
                            user=user,
                            password=password,
                            database=database,
                            cursorclass=pymysql.cursors.DictCursor)
    with connection:
        with connection.cursor() as cursor:
            sql = f'SELECT * FROM {database}.{table} WHERE DateTime="{datetime_str}";'
            cursor.execute(sql)
            result = cursor.fetchall()
            return result

def query_drawing_type(database, table, datetime):
    connection = pymysql.connect(host=host,
                            user=user,
                            password=password,
                            database=database,
                            cursorclass=pymysql.cursors.DictCursor)
    with connection:
        with connection.cursor() as cursor:
            sql = f'SELECT * FROM {database}.{table} WHERE DateTime="{datetime}";'
            cursor.execute(sql)
            result = cursor.fetchall()
            return result


def query_drawing_type_info(database, table, drawing_type):
    connection = pymysql.connect(host=host,
                            user=user,
                            password=password,
                            database=database,
                            cursorclass=pymysql.cursors.DictCursor)
    with connection:
        with connection.cursor() as cursor:
            sql = f'SELECT * FROM {database}.{table} WHERE name="{drawing_type}";'
            cursor.execute(sql)
            result = cursor.fetchall()
            return result

def query_table_sqlite(sqlite_db_path, datetime_str, table):
    connection = sqlite3.connect(sqlite_db_path)
    
    with connection:
        connection.row_factory = dict_factory
        cur = connection.cursor()    
        sql = f'SELECT * FROM {table} WHERE DateTime="{datetime_str}";'
        result = cur.execute(sql).fetchall()
        return result

def query_drawing_type_info_sqlite(sqlite_db_path, table, drawing_type):
    connection = sqlite3.connect(sqlite_db_path)
    
    with connection:
        connection.row_factory = dict_factory
        cursor = connection.cursor()    
        sql = f'SELECT * FROM {table} WHERE name="{drawing_type}";'
        result = cursor.execute(sql).fetchall()
        
        return result

def query_drawing_type_sqlite(sqlite_db_path, table, datetime):
    connection = sqlite3.connect(sqlite_db_path)
    
    with connection:
        connection.row_factory = dict_factory
        cursor = connection.cursor()    
        sql = f'SELECT * FROM {table} WHERE DateTime="{datetime}";'
        cursor.execute(sql)
        result = cursor.fetchall()
        return result

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

# query_table_sqlite(sqlite_db_path, "2020-11-14 10:15:00", "sGroups")
# query_drawing_type_info(sqlite_db_path, 'drawing_type', "USET")
# query_drawing_type(sqlite_db_path, 'drawings', "2020-11-14 10:15:00")


def get_drawing_radius3(database, table, datetime):
    dr_type_query = query_drawing_type_sqlite(database,table, datetime)[0]
    
    d_type = dr_type_query['TypeOfDrawing']
    
    dr_type_info_query = query_drawing_type_info_sqlite(database, 'drawing_type', d_type)[0]
    
    dr_calib_info_query = query_drawing_type_sqlite(database,'calibrations', datetime)[0]
#                             query_drawing_calibration_info(database, 'calibrations', datetime)[0]

    dr_width = dr_type_info_query['width']
    dr_height = dr_type_info_query['height']
    dr_centre_frac_w = dr_type_info_query['pt1_fraction_width']
    dr_centre_frac_h = dr_type_info_query['pt1_fraction_height']
    dr_north_frac_w = dr_type_info_query['pt2_fraction_width']
    dr_north_frac_h = dr_type_info_query['pt2_fraction_height']
    

    dr_center_mm = np.array([dr_width*dr_centre_frac_w, dr_height*dr_centre_frac_h])
    dr_north_mm = np.array([dr_width*dr_north_frac_w, dr_height*dr_north_frac_h])
    
    radius_mm = np.linalg.norm(dr_north_mm - dr_center_mm)
    radius_px = dr_calib_info_query['Radius']
    dr_center_px = np.array([dr_calib_info_query['CenterX'],dr_calib_info_query['CenterY']])
    
    return radius_mm, radius_px, dr_center_mm, dr_center_px

def is_intersect(a, b):
    if a[0] > b[2] or a[2] < b[0]:
#     if self.min_x > other.max_x or self.max_x < other.min_x:
        return False
    if a[1] > b[3] or a[3] < b[1]:
#     if self.min_y > other.max_y or self.max_y < other.min_y:
        return False
    return True

def hasOverlappingBbox(bbox_lst):
    for i in range(len(bbox_lst)):
        a = bbox_lst[i]
        for j in range(i+1,len(bbox_lst)):
            b = bbox_lst[j]
            if is_intersect(a,b):
                return True
    return False


# +
from astropy.wcs.utils import (skycoord_to_pixel, pixel_to_skycoord,
                                _has_distortion, wcs_to_celestial_frame)
def get_sunspots3(h, m, mask, sky_coords=True):
    mask2 = label(mask)
    props_spots = regionprops(mask2)
    
    centers = []
    areas = []
    for prop in props_spots:
#         print(prop.centroid)  # !!!! coordinates are Y,X
        center = list(prop.centroid)
        area = prop.area 
        centers.append(center)
        areas.append(area)

    wcs = WCS(h)
    wcs.heliographic_observer = m.observer_coordinate
    
    centers_arr = np.array(centers)
    sk = None
    
    if len(centers_arr) > 0:
#         centers_arr[:,0]= mask.shape[0]-centers_arr[:,0]
        if sky_coords:
            sk = pixel_to_skycoord(centers_arr[:,1], centers_arr[:,0], wcs, origin=0)
            sk = sk.transform_to(frames.HeliographicCarrington)
        else:
            sk = centers_arr
        
    return sk, areas

def get_sunspots4(h, m, mask, Rmm, sky_coords=True):
    mask2 = label(mask)
    props_spots = regionprops(mask2)
    
    centers = []
    areas = []
    for prop in props_spots:
#         print(prop.centroid)  # !!!! coordinates are Y,X
        center = list(prop.centroid)
        area = prop.area 
        centers.append(center)
        areas.append(area)

    wcs = WCS(h)
    wcs.heliographic_observer = m.observer_coordinate
    
    centers_arr = np.array(centers)
    sk = None
    
    if len(centers_arr) > 0:
#         centers_arr[:,0]= mask.shape[0]-centers_arr[:,0]
        if sky_coords:
            sk = pixel_to_skycoord(centers_arr[:,1], centers_arr[:,0], wcs, origin=0)
            sk = sk.transform_to(frames.HeliographicCarrington)
            ###########
            # When skycoord, then area returned in muHem
            sun_center = np.array([mask.shape[0]//2,mask.shape[1]//2])
            sun_radius = h["SOLAR_R"]
#             print(areas)
#             areas =[get_muHem_area2( Rmm, center, area, sun_radius, sun_center)[0] for area in areas]
            areas =[get_muHem_area(h, Rmm, centers[i], area, sun_center) for i,area in enumerate(areas)]
#             print(areas)
            # filter out sk and areas with area  is nan
            sk = sk[~np.isnan(areas)]
            areas = [area for area in areas if not np.isnan(area)]

            ###########
        else:
            sk = centers_arr

            sun_center = np.array([mask.shape[0]//2,mask.shape[1]//2])
            areas =[get_muHem_area(h, Rmm, centers[i], area, sun_center) for i,area in enumerate(areas)]
            sk = sk[~np.isnan(areas)]
            areas = [area for area in areas if not np.isnan(area)]
        
    return sk, areas


# -

def get_angle(pix_pos, sun_radius, sun_center):
    radius = np.sqrt((pix_pos[0]-sun_center[0])**2 + (pix_pos[1]-sun_center[1])**2)
    ratio = radius/sun_radius
    rho = np.arcsin(ratio)
    rho_deg = np.rad2deg(rho)
    
    return rho

def get_muHem_area(h, Rmm, pix_pos, sunspot_area_pix, sun_center):
    wl_sun_radius = h["SOLAR_R"]

    rho = get_angle(pix_pos, wl_sun_radius, sun_center)
  
    p_side = Rmm / wl_sun_radius # length in mm of a pixel side
    p_area = p_side*p_side # area of a pixel in squared_mm

    sunspot_area_squared_mm = sunspot_area_pix *  p_area # area of sunspot in squared mm 
    sunspot_area_muHem = (sunspot_area_squared_mm *  (10**6)) / (2*np.pi*(Rmm**2)*np.cos(rho))

    return sunspot_area_muHem


def get_angle2(pix_pos, sun_radius, sun_center):
    radius = np.sqrt((pix_pos[0]-sun_center[0])**2 + (pix_pos[1]-sun_center[1])**2)
    ratio = radius/sun_radius
    rho = np.arcsin(ratio)
    rho_deg = np.rad2deg(rho)
    
    return rho

def get_muHem_area2( Rmm, pix_pos, sunspot_area_pix, sun_radius, sun_center):

    rho = get_angle2(pix_pos, sun_radius, sun_center)
  
    p_side = Rmm / sun_radius # length in mm of a pixel side
    p_area = p_side*p_side # area of a pixel in squared_mm

    sunspot_area_squared_mm = sunspot_area_pix *  p_area # area of sunspot in squared mm 
    sunspot_area_muHem = (sunspot_area_squared_mm *  (10**6)) / (2*np.pi*(Rmm**2)*np.cos(rho))

    return sunspot_area_muHem, np.rad2deg(rho)


def grouplist2bboxes_and_rectangles(group_list, drawing_radius_px, wl_radius , pixel_coords ):
    bboxes = []
    bboxes_wl = []
    rectangles = []
    rectangles_wl = []
    for i,item in enumerate(group_list):
        ############ on drawing
        coords = np.array([item['posx'],item['posy']])

        bbox_lon, bbox_lat = c2bb.group_frame2(item["Zurich"], drawing_radius_px,
                                               item["Latitude"], item["Longitude"], 0, 0)

        bbox = [coords[0]-(bbox_lat)/2,
                        coords[1]-(bbox_lon)/2,
                        coords[0]+(bbox_lat)/2,
                        coords[1]+(bbox_lon)/2]

        item["bbox"] = bbox

        bboxes.append(bbox)

        rectangle = plt.Rectangle((bbox[0], bbox[1]),
                                  bbox_lat, bbox_lon ,
                                  color='b', fill=False)

        rectangles.append(rectangle)

        ############ on whitelight  
        coords_wl = pixel_coords[i]
        bbox_wl_lon, bbox_wl_lat = c2bb.group_frame2(item["Zurich"], wl_radius,
                                               item["Latitude"], item["Longitude"], 0, 0)
#         bbox_wl_lon, bbox_wl_lat = c2bb.group_frame2(item["Zurich"], h["SOLAR_R"],
#                                                item["Latitude"], item["Longitude"], 0, 0)

        bbox_wl = [coords_wl[0]-(bbox_wl_lat)/2,
                        coords_wl[1]-(bbox_wl_lon)/2,
                        coords_wl[0]+(bbox_wl_lat)/2,
                        coords_wl[1]+(bbox_wl_lon)/2]

        item["bbox_wl"] = bbox_wl

        bboxes.append(bbox)
        bboxes_wl.append(bbox_wl)

        rectangle_wl = plt.Rectangle((bbox_wl[0], bbox_wl[1]),
                                  bbox_wl_lat, bbox_wl_lon ,
                                  color='b', fill=False)

        rectangles_wl.append(rectangle_wl)
    
    return bboxes, bboxes_wl, rectangles, rectangles_wl


def get_groups_and_radii_in_db(basename, dr_basenames, database):
    datetime = whitelight_to_datetime(basename)
    datetime_str = datetime_to_db_string(datetime)
    closest_dr = find_closest_drawing(datetime_to_drawing_name(datetime), dr_basenames)
#     print(closest_dr)
    
    datetime2 = drawing_name_to_datetime(closest_dr)
    datetime2_str = datetime_to_db_string(datetime2)#.replace('-', ':')

    query_drawing = query_table_sqlite(database, datetime2_str , 'drawings')
    
    angle = query_drawing[0]["AngleP"]
        
    query_group = query_table_sqlite(database, datetime2_str,  'sGroups')

    group_list = [ {
                        "id":item['id'],
                        "Latitude": item["Latitude"],
                        "Longitude": item["Longitude"],
                        "posx": item["PosX"],
                        "posy": item["PosY"],
                        "Lcm": item["Lcm"],
                        "Zurich": item["Zurich"],
                        "McIntosh": item["McIntosh"],
                        "angle": item["CenterToLimbAngle"],
                        "area_px" : item["RawArea_px"],
                        "area_muHem" : item["DeprojArea_msh"]
                    } for item in query_group]
    LonLat_lst = []
    for item in group_list:
        LonLat_lst.append([item["Longitude"],item["Latitude"]])

    LonLat_arr = np.array(LonLat_lst)

    dr_date = datetime2_str
    dr_date = dr_date.replace(' ', 'T') 
    
    wl_date = datetime_str
    wl_date = wl_date.replace(' ', 'T')       

#     drawing_radius_mm, drawing_radius_px = get_drawing_radius2(database, 'drawings', datetime2_str)
    drawing_radius_mm, drawing_radius_px,drawing_center_mm,drawing_center_px = get_drawing_radius3(database, 'drawings', datetime2_str)
    
    
    return group_list, drawing_radius_mm, drawing_radius_px, drawing_center_mm,drawing_center_px, wl_date, dr_date

def wl_list2dbGroups(wl_lst, dr_basenames, database):
    
    out_dict = {}
    
    for wl in tqdm(wl_lst):
        basename = os.path.splitext(os.path.basename(wl))[0]
        
        group_list, drawing_radius_mm,drawing_radius_px,drawing_center_mm,drawing_center_px,  wl_date, dr_date = get_groups_and_radii_in_db(basename, dr_basenames, database)
        
        # print(group_list)

        out_dict[basename]= {
                                "group_list" : group_list,
                                "dr_radius_mm": drawing_radius_mm,
                                "dr_radius_px": drawing_radius_px,
                                "dr_center_mm": drawing_center_mm,
                                "dr_center_px": drawing_center_px,
                                "wl_date": wl_date,
                                "dr_date": dr_date,
                            }

        
    return out_dict


def get_unique_drawing_datetimes(database, table):
    connection = sqlite3.connect(database)
    
    with connection:
#         connection.row_factory = dict_factory
        cursor = connection.cursor()  
        sql = f'SELECT DISTINCT DateTime FROM {table};'
        result = cursor.execute(sql).fetchall()
        return result

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

def expand_small_spots( msk):
        out_msk = msk.copy()
        label_img = label(out_msk)
        regions = regionprops(label_img)
        
        for r in regions:
            if r.area == 1:
                coords = r.coords[0]
                # print(coords)
                out_msk[coords[0]-1:coords[0]+1,coords[1]-1:coords[1]+1] = msk[coords[0],coords[1]]
                
        return out_msk

def rotate_img_opencv(image, angle, interpolation):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=interpolation)
    return rotated

def rotate_CV_bound(image, angle, interpolation):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH),flags=interpolation)


def get_mask_from_coords(mask, coords):
        m = np.zeros_like(mask)
        m = m.astype(np.uint8)
        m2 = mask.copy()
        m2[m2>0] = 1
        l = label(m2)
        # print(np.unique(l)[1:])
        for val in np.unique(l)[1:]:
            # Get contours
            contours, hierarchies = cv2.findContours((l==val).astype(np.uint8) , cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Get convex hulls of contours to avoid missing sunspots with exotic shapes
            hull_list = []
            for i in range(len(contours)):
                hull = cv2.convexHull(contours[i])
                hull_list.append(hull)

            # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            # ax[0].imshow(l==val.astype(np.uint8))
            # for c in contours:
            #     ax[0].plot(c[:,:,0], c[:,:,1], c='r')
            # plt.show()
            # print(contours)

            # print(coords)
            # Check if any of the coordinates is inside the convex hulls
            for c in coords:
                for cnt in hull_list:

                    # If the point is inside the convex hull, add corresponding sunspot to mask
                    if cv2.pointPolygonTest(cnt, (c[1], c[0]), False) >= 0:
                        m += (l==val).astype(np.uint8)
                        break


        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(mask)
        # ax[0].scatter([c[1] for c in coords], [c[0] for c in coords], c='r', s=10)
        # ax[1].imshow(m)
        # plt.show()

        return m

import MeanShift as ms

def process_one_image(wl,
                      huge_db_dict, 
                      huge_dict,
                      wl_list, rotten_list,
                      masks_dir,
                      look_distance, kernel_bandwidthLon, kernel_bandwidthLat, n_iterations,
                      input_type # "mask" or "confidence map"
                      ):

    assert input_type in ["mask", "confidence_map"]

    basename = os.path.splitext(os.path.basename(wl))[0]
    if( basename in list(huge_dict.keys())) or (basename.startswith("UCC")
        or ( wl_list.index(wl) in rotten_list)):
#         continue
        return basename, {}

    try: 
        
        cur_db_dict = huge_db_dict[basename]
        # print(cur_db_dict)
        
        group_list = cur_db_dict["group_list"]

        # print('process_one_image: ', group_list)

        drawing_radius_mm = cur_db_dict["dr_radius_mm"]
        drawing_radius_px = cur_db_dict["dr_radius_px"]
        date2 = cur_db_dict["dr_date"]
        date = cur_db_dict["wl_date"] 

        # print('process_one_image: ',date, date2)
        
        #####################
        m, h = utils.open_and_add_celestial(wl)
        corrected = False
        if not 'DATE-OBS' in h:
            # print('corrected')
            m, h = utils.open_and_add_celestial2(wl, date_obs=date)
            corrected = True


        # pre-2005 images have their data image flipped vertically

        flip_time = "2003-03-08T00:00:00"
        should_flip = (datetime.fromisoformat(date) - datetime.fromisoformat(flip_time)) < timedelta(0)
        if should_flip:
            m = sunmap.Map(np.flip(m.data,axis=0), h)



    #     m, h = utils.open_and_add_celestial2(wl_list[tmp_idx], date_obs=date)
        wcs = WCS(h)
        wcs.heliographic_observer = m.observer_coordinate
        origin = m.data.shape[0]//2, m.data.shape[1]//2
        
        # On a besoin des coordonnées pixel des bboxes après correction de l'angle Solaire ->> on doit 
        # calculer un wcs transformé 
        m_rot = m.rotate(angle=-h["SOLAR_P0"] * u.deg)
        top_right = SkyCoord( 1000 * u.arcsec, 1000 * u.arcsec, frame=m_rot.coordinate_frame)
        bottom_left = SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=m_rot.coordinate_frame)
        m_rot_submap = m_rot.submap(bottom_left, top_right=top_right)
        m_rot_submap_shape = m_rot_submap.data.shape
        m_rot_shape = m_rot.data.shape
        deltashapeX = np.abs(m_rot_shape[0]-m_rot_submap_shape[0])
        deltashapeY = np.abs(m_rot_shape[1]-m_rot_submap_shape[1]) 

        h2 = m_rot_submap.fits_header 
        h2.append(('CTYPE1', 'HPLN-TAN'))
        h2.append(('CTYPE2', 'HPLT-TAN'))
        wcs2 = WCS(h2)
        wcs2.heliographic_observer = m_rot_submap.observer_coordinate
        origin = m_rot_submap.data.shape[0]//2, m_rot_submap.data.shape[1]//2
        #####################
        dr_obstime = date+'.000'  
        all_sks = []
        all_pixels = []
        for item in group_list:
            cur_sk = SkyCoord(item["Longitude"]*u.rad, item["Latitude"]*u.rad , frame=frames.HeliographicCarrington,
                        obstime=dr_obstime, observer="earth") 
            coords_wl = skycoord_to_pixel(cur_sk, wcs2, origin=0)
            all_sks.append(cur_sk)
            all_pixels.append(coords_wl)
        # if should_flip:
        #     # all_pixels = [(m_rot_submap.data.shape[0]-p[0], p[1]) for p in all_pixels]
            # all_pixels = [(p[0], m_rot_submap.data.shape[1]-p[1]) for p in all_pixels]
            # all_pixels = [(m_rot_submap.data.shape[0]-p[0]+deltashapeX, p[1]) for p in all_pixels]


        # print('before',group_list)
        bboxes, bboxes_wl, rectangles, rectangles_wl = grouplist2bboxes_and_rectangles(group_list, 
                                                                                    drawing_radius_px,
                                                                                    h2["SOLAR_R"],
                                                                                    all_pixels)   
        # print('after',group_list)
        
        basename2 = os.path.splitext(os.path.basename(wl))[0]
        if input_type == "mask":
            mask = io.imread(os.path.join(masks_dir,basename2+".png"))
        elif input_type == "confidence_map":
            mask = np.load(os.path.join(masks_dir,basename2+"_proba_map.npy"))
            # raise AssertionError("Not implemented yet")
        mask[mask>0] = 1

        # print('should_flip', should_flip)
        if should_flip:
            mask = np.flip(mask,axis=0)
            disp_mask = rotate_CV_bound(mask, angle=h["SOLAR_P0"], interpolation=cv2.INTER_NEAREST) #rotate(mask, angle=h["SOLAR_P0"], reshape=True)     
        else:       
            disp_mask = rotate_CV_bound(mask, angle=h["SOLAR_P0"], interpolation=cv2.INTER_NEAREST) #rotate(mask, angle=h["SOLAR_P0"], reshape=True)
        
        disp_mask = disp_mask[deltashapeX//2:disp_mask.shape[0]-deltashapeX//2,
                            deltashapeY//2:disp_mask.shape[1]-deltashapeY//2] 
        cur_key = basename
        cur_dict = {}
        cur_dict['db'] = group_list

        sunspots_pixel, _ = get_sunspots3(h2, m_rot_submap, disp_mask, sky_coords=False)
        sunspots_sk, sunspots_areas = get_sunspots3(h2, m_rot_submap, disp_mask, sky_coords=True)    
        # print("sunspots_pixel",sunspots_pixel)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection=m_rot_submap)
        # m_rot_submap.plot(axes=ax, title="Sunspots")
        # m_rot_submap.draw_grid(axes=ax)
        # if sunspots_pixel is not None:
        #     ax.scatter(sunspots_pixel[:,1],sunspots_pixel[:,0], color='red', s=10)
        # # plt.show()
        
        # fig, ax = plt.subplots(nrows=1,ncols=1)
        # ax.imshow(disp_mask)
        # if sunspots_pixel is not None:
        #     ax.scatter(sunspots_pixel[:,1],sunspots_pixel[:,0], color='red', s=10)
        

    #     sunspots_pixel, _ = get_sunspots3(h, m, mask, sky_coords=False)
    #     sunspots_sk, sunspots_areas = get_sunspots3(h,m, mask, sky_coords=True)
    #         print(sunspots_sk.radius.km[0])

        ms_centroids = []
        ms_group_sunspots = []
        ms_group_areas = []
        
        
        if sunspots_pixel is None:
            cur_dict['SOLAR_P0'] = h["SOLAR_P0"]
            cur_dict['deltashapeX'] = deltashapeX
            cur_dict['deltashapeY'] = deltashapeY
            
            cur_dict['seg'] = {}
            cur_dict['meanshift'] = { 
                                        "centroids": [],"groups": [],"areas": [],
                                        "centroids_px": [],"groups_px": [],
                                    }
            cur_dict['kmeans'] = {"centroids": [],"groups": [],"areas": []}
        else:
            sk_Lon = sunspots_sk.lon.rad
            sk_Lat = sunspots_sk.lat.rad
            sk_LatLon = np.stack((sk_Lat,sk_Lon),axis=1)
        
            nan_indexes = np.unique(np.argwhere(np.isnan(sk_LatLon))[:,0])
            clean = (~np.isnan(sk_Lon) & ~np.isnan(sk_Lat))
        #     print(clean)
            if len(nan_indexes) > 0:
                sunspots_sk = sunspots_sk[clean]
                sunspots_pixel = sunspots_pixel[clean]
                sunspots_areas = (np.array(sunspots_areas)[clean]).tolist()
                sk_LatLon = sk_LatLon [clean]

    #         print("sk_LatLon", sk_LatLon)

            if len(sk_LatLon)==0:
                cur_dict['SOLAR_P0'] = h["SOLAR_P0"]
                cur_dict['deltashapeX'] = deltashapeX
                cur_dict['deltashapeY'] = deltashapeY
                
                cur_dict['seg'] = {}
                cur_dict['meanshift'] = { 
                                        "centroids": [],"groups": [],"areas": [],
                                        "centroids_px": [],"groups_px": [],
                                        }
                cur_dict['kmeans'] = {"centroids": [],"groups": [],"areas": []}
            
            else:    
            

                sun_center = np.array([disp_mask.shape[0]//2,disp_mask.shape[1]//2])
    #             sun_center = np.array([mask.shape[0]//2,mask.shape[1]//2])
                Rmm = drawing_radius_mm
    #             sunspots_areas_mmHem =[get_muHem_area(h, Rmm, sunspots_pixel[i], sunspots_areas[i],sun_center)
    #                                                  for i, sk in enumerate(sunspots_sk)]
                try:
                    sunspots_areas_mmHem =[get_muHem_area(h2, Rmm, sunspots_pixel[i], sunspots_areas[i],sun_center)
                                                    for i, sk in enumerate(sunspots_sk)]
                except IndexError:
                    print(wl)
                    raise(IndexError)

                ############ Mean-Shift

                ms_model = ms.Mean_Shift(look_distance, kernel_bandwidthLon, kernel_bandwidthLat, sunspots_sk.radius.km[0], n_iterations,
                                        max_scaled_area_muHem=200)
                ms_model.fit(sk_LatLon, sunspots_areas_mmHem)
    #             ms_model.fit(sk_LatLon,sunspots_areas)
                ms_centroids = ms_model.centroids

                try:
                    sk_sequ_meanshift = SkyCoord(ms_centroids[:,1]*u.rad, ms_centroids[:,0]*u.rad , frame=frames.HeliographicCarrington,
                                obstime=m.date, observer="earth")
                except IndexError:
                    print(f'{wl} index: {wl_list.index(wl)}')
                    raise IndexError

    #             pix_centers_meanshift = skycoord_to_pixel(sk_sequ_meanshift, wcs, origin=0)
                pix_centers_meanshift = np.array(skycoord_to_pixel(sk_sequ_meanshift, wcs2, origin=0)).T
                # if should_flip:
                #     pass
                #     # print("should flip2")
                #     # print("pix_centers_meanshift", pix_centers_meanshift)
                #     pix_centers_meanshift[:,0] = disp_mask.shape[0] - pix_centers_meanshift[:,0]
                #     # pix_centers_meanshift[:,1] = disp_mask.shape[1] - pix_centers_meanshift[:,1]
                #     # print("pix_centers_meanshift", pix_centers_meanshift)
                pix_centers_meanshift = pix_centers_meanshift.tolist()

    #             print("pix_centers_meanshift", pix_centers_meanshift)
            #     print("Mean_shift centroids", ms_centroids)

                ms_classifications = ms_model.predict(sk_LatLon)
            #     print("Mean sift Classifications", ms_classifications)

                ms_group_sunspots = [(sk_LatLon[ms_classifications == c].tolist()) for c in np.unique(ms_classifications)]
                # if should_flip:
                #     # print("should flip3")
                #     # print("sunspots_pixel", sunspots_pixel)
                #     sunspots_pixel[:,1] = disp_mask.shape[0] - sunspots_pixel[:,1]
                #     # print("sunspots_pixel", sunspots_pixel)
                ms_group_sunspots_px = [sunspots_pixel[ms_classifications == c].tolist() for c in np.unique(ms_classifications)]


    #             print("ms_group_sunspots_px" , ms_group_sunspots_px)
                ms_group_areas = [ np.sum(np.array(sunspots_areas_mmHem)[ms_classifications == c])
                                                            for c in np.unique(ms_classifications)]

                cur_dict['SOLAR_P0'] = h["SOLAR_P0"]
                cur_dict['deltashapeX'] = deltashapeX
                cur_dict['deltashapeY'] = deltashapeY
            
                cur_dict['seg'] = {i: {'pos': sk.tolist(), "area": str(sunspots_areas[i])}  for i, sk in enumerate(sk_LatLon) }
                cur_dict['meanshift'] = { "centroids": ms_centroids.tolist(), 
                                            "groups": ms_group_sunspots,
                                            "areas": ms_group_areas,
                                            "centroids_px": pix_centers_meanshift, 
                                            "groups_px": ms_group_sunspots_px,  
                                        }
    except:
        print(f'error in {basename}')
        raise
    
    return cur_key, cur_dict



def grouplist2bboxes(group_list, drawing_radius_px):
    bboxes = []
    for i,item in enumerate(group_list):
        ############ on drawing
        coords = np.array([item['posx'],item['posy']])

        bbox_lon, bbox_lat = c2bb.group_frame2(item["Zurich"], drawing_radius_px,
                                               item["Latitude"], item["Longitude"], 0, 0)

        bbox = [coords[0]-(bbox_lat)/2,
                        coords[1]-(bbox_lon)/2,
                        coords[0]+(bbox_lat)/2,
                        coords[1]+(bbox_lon)/2]

        bboxes.append(bbox)
    
    return bboxes



def contains_sunspot(db_bbox, sunspot):
    # function that checks if a sunspot is in a bbox

    if  (db_bbox[0] <= sunspot[1] <= db_bbox[2]) and (
        db_bbox[1] <= sunspot[0] <= db_bbox[3]):
            return True
    return False

def contains_group(db_bbox, group ):
    # function that checks if a group of sunspots is in a bbox
    intersect = [contains_sunspot(db_bbox, sunspot) for sunspot in group]   
    if any(intersect):
        return True
    return False

def contains_ms_groups(db_bbox, db_center, ms_centers, ms_groups ):  
    # function that checks that the bbox contains groups of sunspots
    # returns a list of booleans, one for each group of sunspots.
    # Sum the list to get the number of groups in the bbox
    intersect = [contains_group(db_bbox, group) for group in ms_groups]
    return intersect

def get_intersecting_db_bboxes(db_bboxes):
    # function that returns the number of intersections between bboxes
    # returns a list of integers, one for each bbox.
    
    intersections = np.zeros((len(db_bboxes), len(db_bboxes)))
#     intersections = [0]*len(db_bboxes)
    
    for i in  range(len(db_bboxes)-1):
        for j in range(i+1,len(db_bboxes)):
            bbox,bbox2 = db_bboxes[i],db_bboxes[j]
            if is_intersect(bbox,bbox2):
                intersections[i,j] = 1
                intersections[j,i] = 1
                
    
    return np.sum(intersections, axis=0)

# function that counts the number of bboxes a group of sunspots intersects
def count_group_intersections(group, db_bboxes):
    # function that returns the number of intersections between given group and bboxes
    # returns a list of integers, one for each bbox.

    # print(group)
    # print(db_bboxes)

    num_sunspots = len(group)
    intersections = np.zeros((num_sunspots, len(db_bboxes)))

    for i in  range(num_sunspots):
        for j in  range(len(db_bboxes)):
            # print(group[i], db_bboxes[j])
            if contains_sunspot(db_bboxes[j], group[i]):
                intersections[i,j] = 1
    # print(intersections)
    
    sums = np.sum(intersections, axis=0)

    # print(sums)

    out = sums > 0

    return out


def get_bbox_from_mask(mask):
    '''returns the bounding box of a mask.
    mask: 2D array of booleans
    returns: [x0,y0,x1,y1]'''
    # Attention: x represents Latitude and y represents Longitude
    # The given mask must have been rotated so that solar rotation axis is vertical
    x,y = np.where(mask)
    return np.min(x), np.min(y), np.max(x), np.max(y)




