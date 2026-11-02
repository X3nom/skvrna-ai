import os
import numpy as np
import glob
import json

import matplotlib.patches as patches

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle

from sunpy.coordinates import frames
import sunpy.map as sunmap

import skimage.io as io

import MeanShift as MS

from copy import deepcopy
import pandas as pd

import clustering_utilities as c_utils
import tracking_utilities as t_utils

import importlib
importlib.reload(c_utils)

import ipywidgets as widgets
import matplotlib.pyplot as plt
import seaborn as sns

import panel as pn

from sklearn.model_selection import ParameterGrid
import concurrent.futures
from itertools import repeat
import multiprocessing

from tqdm.notebook import tqdm

from datetime import datetime, timedelta


#######  Counting Functions to get number of matchings


import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def compute_IoUs(db_bboxes, ms_bboxes):
    '''Compute IoUs between all bboxes in db_bboxes and ms_bboxes.'''
    db_bboxes = np.array(db_bboxes)
    ms_bboxes = np.array(ms_bboxes)
    # print(db_bboxes.shape)
    # print(ms_bboxes.shape)

#     print('compute_iou, ', db_bboxes.shape, ms_bboxes.shape)
#     print('compute_iou, ', db_bboxes.ndim, ms_bboxes.ndim)
    

    # if db_bboxes is single bbox, convert to 2D array.
    if db_bboxes.ndim == 1:
        db_bboxes = np.array([db_bboxes])
    # if ms_bboxes is single bbox, convert to 2D array.
    if ms_bboxes.ndim == 1:
        ms_bboxes = np.array([ms_bboxes])

    # if db_bboxes is empty, return empty 2D array.
    if db_bboxes.size == 0:
        return np.array([[]])
    # if ms_bboxes is empty, return empty 2D array.
    if ms_bboxes.size == 0:
        return np.array([[]])

    # print(db_bboxes.shape)
    # print(ms_bboxes.shape)
    # db_bboxes has shape (N, 4) and ms_bboxes has shape (M, 4).
    # each element has the form [x1, y1, x2, y2].
    # We want to compute IoUs between all N and M boxes.
    db_bboxes = db_bboxes.astype(np.float)
    ms_bboxes = ms_bboxes.astype(np.float)
    # print(db_bboxes.shape)
    # print(ms_bboxes.shape)

    # Compute areas of all db_bboxes and ms_bboxes.
    area_db = (db_bboxes[:, 2] - db_bboxes[:, 0]) * (db_bboxes[:, 3] - db_bboxes[:, 1])
    area_ms = (ms_bboxes[:, 2] - ms_bboxes[:, 0]) * (ms_bboxes[:, 3] - ms_bboxes[:, 1])
    # print(area_db.shape, area_ms.shape)

    # compute intersections
    # intersections has shape (N, M) and intersections[i, j] is the intersection
    # between db_bboxes[i] and ms_bboxes[j].
    intersections = np.zeros((db_bboxes.shape[0], ms_bboxes.shape[0]))
    for i in range(db_bboxes.shape[0]):
        for j in range(ms_bboxes.shape[0]):
            x1 = max(db_bboxes[i, 0], ms_bboxes[j, 0])
            y1 = max(db_bboxes[i, 1], ms_bboxes[j, 1])
            x2 = min(db_bboxes[i, 2], ms_bboxes[j, 2])
            y2 = min(db_bboxes[i, 3], ms_bboxes[j, 3])
            intersections[i, j] = max(x2 - x1, 0) * max(y2 - y1, 0)
    # print(intersections.shape)
    # print(intersections)

    # compute unions
    unions = area_db[:, np.newaxis] + area_ms[np.newaxis, :] - intersections

    # compute IoUs
    ious = intersections / unions


    return ious

def compute_distances(db_bboxes, ms_bboxes):
    '''Compute distances between all bboxes in db_bboxes and ms_bboxes.'''

    db_bboxes = np.array(db_bboxes)
    ms_bboxes = np.array(ms_bboxes)

    # if db_bboxes is single bbox, convert to 2D array.
    if db_bboxes.ndim == 1:
        db_bboxes = np.array([db_bboxes])
    # if ms_bboxes is single bbox, convert to 2D array.
    if ms_bboxes.ndim == 1:
        ms_bboxes = np.array([ms_bboxes])

    # if db_bboxes is empty, return empty 2D array.
    if db_bboxes.size == 0:
        return np.array([[]])
    # if ms_bboxes is empty, return empty 2D array.
    if ms_bboxes.size == 0:
        return np.array([[]])

    # db_bboxes has shape (N, 4) and ms_bboxes has shape (M, 4).
    # each element has the form [x1, y1, x2, y2].
    # We want to compute distances between all N and M boxes.
    db_bboxes = db_bboxes.astype(np.float)
    ms_bboxes = ms_bboxes.astype(np.float)

    # Compute centers of all db_bboxes and ms_bboxes.
    center_db = (db_bboxes[:, 2:] + db_bboxes[:, :2]) / 2
    center_ms = (ms_bboxes[:, 2:] + ms_bboxes[:, :2]) / 2
    # print('here')
    # print(center_db.shape)
    # print(center_ms.shape)

    distances = np.repeat( center_ms[ np.newaxis,:,:], center_db.shape[0], axis=0)

    # print(distances.shape)

    distances = np.sqrt(np.sum((distances - center_db[:,np.newaxis,:])**2, axis=2))

    # print(distances.shape)
    # print(distances)

    return distances

def no_ms_at_all(db_bboxes):
    '''If there is no ms at all, return empty list.'''
    
    ious =  np.array([[0] for i in range(len(db_bboxes))])
    distances =  np.array([[np.Inf] for i in range(len(db_bboxes))])
    unmatched_ms =  np.array([])
    bad_ms =  np.array([])
    ms_too_far_idx =  np.array([])
    unmatched_db =  np.array([range(len(db_bboxes))])
    multiDB_singleMS_idx =  np.array([])
    db_too_far_idx = np.array([range(len(db_bboxes))])
    matched_db =  np.array([])
    matches =  np.array([])

    return  (ious, distances, 
            unmatched_ms, bad_ms, ms_too_far_idx, 
            unmatched_db, multiDB_singleMS_idx, db_too_far_idx, 
            matched_db, matches)


def find_closest_ms_bbox(db_bboxes, ms_bboxes, maximum_distance=300):
    '''Find the closest ms_bbox to db_bbox.'''
    
    if len(ms_bboxes) == 0 and len(db_bboxes) > 0:
        return no_ms_at_all(db_bboxes)


    ious = compute_IoUs(db_bboxes, ms_bboxes)
    distances = compute_distances(db_bboxes, ms_bboxes)

#     print("ious", ious)
#     print("distances", distances)

    

    # if ious is empty, closest_ms_bbox is empty.
    if (ious.size == 0) and (distances.size == 0):
        # print("find_closest_ms_bbox: ious and distances are empty.")
        closest_ms_bbox = np.array([])
        closest_ms_bbox_idx = np.array([])
        
        closest_db_bbox = np.array([])
        closest_db_bbox_idx = np.array([])
    else:
        # Find the closest ms_bbox to each db_bbox.
        closest_ms_bbox = np.min(distances, axis=1)
        closest_ms_bbox_idx = np.argmin(distances, axis=1)
        # print("closest_ms_bbox_idx", closest_ms_bbox_idx)
        
        
        # Find the closest db_bbox to each ms_bbox.
        closest_db_bbox = np.min(distances.T, axis=1)
        closest_db_bbox_idx = np.argmin(distances.T, axis=1)
    
    # print(closest_ms_bbox.shape)
    # print(closest_ms_bbox_idx.shape)
    # print(closest_ms_bbox)
    # print(closest_ms_bbox_idx)

    # replace the closest ms_bbox with -1 if it is too far away.
    closest_ms_bbox[closest_ms_bbox > maximum_distance] = -1
    closest_ms_bbox_idx[closest_ms_bbox == -1 ] = -1
     
    # replace the closest db_bbox with -1 if it is too far away.
    closest_db_bbox[closest_db_bbox > maximum_distance] = -1
    closest_db_bbox_idx[closest_db_bbox == -1 ] = -1

    # count the number of db_bboxes that are too far away.
    db_too_far = closest_ms_bbox == -1
    num_db_too_far = np.sum(db_too_far)
    db_too_far_idx = np.where(db_too_far)[0]
    # count the number of ms_bboxes that are too far away.
    ms_too_far = closest_db_bbox == -1
    num_ms_too_far = np.sum(ms_too_far)
    ms_too_far_idx = np.where(ms_too_far)[0]
    
#     print("closest_ms_bbox", closest_ms_bbox)
#     print("closest_ms_bbox_idx", closest_ms_bbox_idx)
    # get the ious of the closest ms_bbox to each db_bbox.
    closest_ms_bbox_iou = np.array([ious[i,closest_ms_bbox_idx[i]] for i in range(len(db_bboxes))])
    closest_ms_bbox_iou[closest_ms_bbox == -1] = -1


    multiDB_singleMS_idx = []
    num_multiDB_singleMS = 0
    # if two db_bboxes have the same closest ms_bbox, keep the one with the highest IoU.
    for i in np.unique(closest_ms_bbox_idx):
        if i == -1:
            continue
        # get the indexes of the db_bboxes that have the same closest ms_bbox.
        idx = np.where(closest_ms_bbox_idx == i)[0]
        if len(idx) > 1:
            # keep the db_bbox with the highest IoU.
            highest_iou_idx = np.argmax(closest_ms_bbox_iou[idx])
            # add the indexes of the other db_bboxes to multiDB_singleMS.
            multiDB_singleMS_idx.extend(idx[np.arange(len(idx)) != highest_iou_idx])

            # set the closest ms_bbox of the other db_bboxes to -1.
            closest_ms_bbox_idx[idx[np.arange(len(idx)) != highest_iou_idx]] = -1
            closest_ms_bbox[idx[np.arange(len(idx)) != highest_iou_idx]] = -1
            closest_ms_bbox_iou[idx[np.arange(len(idx)) != highest_iou_idx]] = -1
            # get the number of indexes that were set to -1.
            num_multiDB_singleMS += len(idx) - 1
    assert num_multiDB_singleMS == len(multiDB_singleMS_idx)
    multiDB_singleMS_idx = np.array(multiDB_singleMS_idx)
    

    # get the indexes of the ms_bboxes that appear as the closest ms_bbox to some db_bbox.
    ms_bbox_idx = np.unique(closest_ms_bbox_idx)
    
    # remove the -1 index.
    ms_bbox_idx = ms_bbox_idx[ms_bbox_idx != -1]
    # print("ms_bbox_idx", ms_bbox_idx)

    # print(closest_ms_bbox, closest_ms_bbox_idx)
    # print(ms_bbox_idx)
    
    candidates_indexes = np.array(range(len(ms_bboxes)))
    unmatched_ms = np.setdiff1d(candidates_indexes, ms_bbox_idx)

    if (ious.size == 0) and (distances.size == 0):
        bad_ms = np.array([])
    else:   
        # get the ms_bboxes in unmatched_ms that have iou > 0. with some db_bbox.
        bad_ms = unmatched_ms[np.max(ious[:, unmatched_ms], axis=0) > 0.]

    # remove bad_ms from unmatched_ms.
    unmatched_ms = np.setdiff1d(unmatched_ms, bad_ms)

    # unmatched_db contains: 
    # 1) the indexes of the db_bboxes that are too far away from any ms_bbox. 
    # 2) the indexes of the db_bboxes that have the same closest ms_bbox as another db_bbox but that have a lower IoU.
    unmatched_db = np.where(closest_ms_bbox == -1)[0]

    # print("unmatched_db", unmatched_db)
    # print("multiDB_singleMS", multiDB_singleMS)
    # print("db_too_far", db_too_far)


    # 2 types of 'rejected'  db bboxes (candidate too far + candidate better matched by another db bbox) : 
    # Make sure that the number of unmatched db + multiDB_singleMS is equal to the number of too far db.
    assert len(unmatched_db) == num_db_too_far + num_multiDB_singleMS
#     assert len(unmatched_ms) == num_ms_too_far +
    
    # Make sure that the number of unmatched db + ms_bbox_idx (number of 1 to 1 matches) is equal to the number of db_bboxes.
    assert len(unmatched_db) + len(ms_bbox_idx) == len(db_bboxes)
    # Make sure that the number of bad ms + unmatched ms + ms_bbox_idx is equal to the number of ms_bboxes.
    assert len(unmatched_ms) + len(bad_ms) + len(ms_bbox_idx) == len(ms_bboxes)

    matched_db = np.where(closest_ms_bbox != -1)[0]
    matches = closest_ms_bbox_idx

    
    return  (ious, distances, 
            unmatched_ms, bad_ms, ms_too_far_idx, 
            unmatched_db, multiDB_singleMS_idx, db_too_far_idx, 
            matched_db, matches)

def find_matchings_one_image(cur_huge_dict, huge_db_dict, basename, wl_dir, masks_dir, input_type, show=False):
        cur_image_dict = cur_huge_dict[basename]

        
        angle = cur_image_dict["SOLAR_P0"]
        deltashapeX = cur_image_dict["deltashapeX"]
        deltashapeY = cur_image_dict["deltashapeY"]
        
        drawing_radius_px = huge_db_dict[basename]["dr_radius_px"]

        date = huge_db_dict[basename]["wl_date"] 
        
        group_list = cur_image_dict['db']
        
        ms_dict = cur_image_dict['meanshift']

        ms_members = ms_dict['groups_px']

        # print('ms_dict: ', ms_dict)
        
        centroids = np.array(ms_dict["centroids"])
        centroids_px = np.array(ms_dict["centroids_px"])
        
        db_classes = [{"Zurich":item['Zurich'], "McIntosh":item['McIntosh'] } for item in group_list]
        # Attention: bbox_wl is in the form [lat1, Lon1, lat2, Lon2] -> [y1, x1, y2, x2]
        # x1 is 
        db_bboxes = [np.array(item['bbox_wl']) for item in group_list]
        db_centers_px = np.array([[(b[2]+b[0])/2,(b[3]+b[1])/2] for b in db_bboxes])

        # open the image
        image = np.array(io.imread(os.path.join(wl_dir, basename + '.FTS')))
        image = c_utils.rotate_CV_bound(image, angle, interpolation=cv2.INTER_NEAREST)
        image = image[deltashapeX//2:image.shape[0]-deltashapeX//2,
                            deltashapeY//2:image.shape[1]-deltashapeY//2]

        # open the mask
        # mask = np.array(io.imread(os.path.join(masks_dir, basename + '.png')))
        if input_type == "mask":
            mask = io.imread(os.path.join(masks_dir,basename+".png"))
        elif input_type == "confidence_map":
#             print("here")
            mask = np.load(os.path.join(masks_dir,basename+"_proba_map.npy"))
            mask[mask>0] = 1   

        flip_time = "2003-03-08T00:00:00"
        should_flip = (datetime.fromisoformat(date) - datetime.fromisoformat(flip_time)) < timedelta(0)
        if should_flip: 
            image = np.flip(image,axis=0)
            mask = np.flip(mask,axis=0)
                
        msk = c_utils.expand_small_spots(mask)


        # rotate the mask
        mask = c_utils.rotate_CV_bound(mask, angle, interpolation=cv2.INTER_NEAREST)
        mask = mask[deltashapeX//2:mask.shape[0]-deltashapeX//2,
                            deltashapeY//2:mask.shape[1]-deltashapeY//2] 

        
        group_masks = [c_utils.get_mask_from_coords(mask, members) for members in ms_dict['groups_px']]
         
        
        groups_bboxes = [c_utils.get_bbox_from_mask(mask) for mask in group_masks]
        groups_bboxes = [(b[1], b[0], b[3], b[2]) for b in groups_bboxes]

        # # print('here - 1')
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # ax.set_title(angle)
        # ax.imshow(image)
        # ax.imshow(mask, alpha=.5)
        # for members in ms_dict['groups_px']:
        #     members = np.array(members)
        #     # print("members", members)
        #     ax.scatter(members[:,1], members[:,0], s=1)

        # for i, bbox in enumerate(db_bboxes):
        #     linestyle = '-'
        #     ax.add_patch(patches.Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],linewidth=1,edgecolor='b',facecolor='none', linestyle=linestyle))
        # for i, bbox in enumerate(groups_bboxes):
        #     color = 'g'
        #     ax.add_patch(patches.Rectangle((bbox[0],bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],linewidth=1,edgecolor=color,facecolor='none'))
                    

        # plt.show()
        
        # print("db_bboxes", db_bboxes)
        # print("groups_bboxes", groups_bboxes)

        res = find_closest_ms_bbox(db_bboxes, groups_bboxes)
        ious, distances, unmatched_ms, bad_ms, ms_too_far, unmatched_db, multiDB_singleMS, db_too_far, matched_db, matches = res
        
        unmatched_ms = unmatched_ms.tolist()
        unmatched_db = unmatched_db.tolist()
        bad_ms = bad_ms.tolist()
        multiDB_singleMS = multiDB_singleMS.tolist()
        db_too_far = db_too_far.tolist()
        ms_too_far = ms_too_far.tolist()
        

        cur_out_stats = {
            # General info
            'num_DB_groups':len(db_bboxes),
            'num_MS_groups':len(centroids_px),

            'matches':matches,

            # MS with DB matching info
            'unmatched_db':unmatched_db,
            'multiDB_singleMS': multiDB_singleMS,
            'db_too_far':db_too_far,


            'unmatched_ms':unmatched_ms,
            'bad_ms':bad_ms,
            'ms_too_far': ms_too_far,

            "ious":ious.tolist(),
            "distances":distances.tolist()

            }
        
        Rmm = huge_db_dict[basename]['dr_radius_mm']
        R_pixel = huge_db_dict[basename]['dr_radius_px']
        sun_center = huge_db_dict[basename]['dr_center_px']
        
        cur_out_groups = []
        for i, match in enumerate(matches):
            # print('i: ', i, 'match: ', match)
            if match != -1:
                db_class = db_classes[i]

                for pt in ms_members[match]:
                    # print('pt: ', pt)
                    assert c_utils.contains_sunspot(groups_bboxes[match],pt), "pt: {} not in bbox: {}".format(pt, groups_bboxes[match])


                dr_pixpos = np.array([group_list[i]['posx'], group_list[i]['posy']])
                
                angular_excentricity =  c_utils.get_angle2(dr_pixpos, R_pixel, sun_center)
                
                cur_group_dict={
                                "centroid_px": centroids_px[match],
                                "centroid_Lat": centroids[match][0],
                                "centroid_Lon": centroids[match][1],
                                "angular_excentricity_rad": angular_excentricity,
                                "angular_excentricity_deg": np.rad2deg(angular_excentricity),
                                "Zurich":   db_class["Zurich"],
                                "McIntosh": db_class["McIntosh"],
                                "members": ms_members[match],
                                "members_mean_px": np.mean(ms_members[match], axis=0),
                            }
                
                
                cur_out_groups.append(cur_group_dict)



        out_groups = {}
        if len(cur_out_groups) > 0:
            out_groups = { "angle": angle,
                                        "deltashapeX":deltashapeX,
                                        "deltashapeY":deltashapeY,
                                        "groups": cur_out_groups,
                                    }


        ############## SHOW THE RESULTS ################
        if show:
            print('unmatched db: ', unmatched_db)
            print('multiDB_singleMS: ', multiDB_singleMS)
            print('db_too_far: ', db_too_far)

            print('unmatched ms', unmatched_ms)
            print('bad_ms: ', bad_ms)

            # if ious in not empty
            if ious.size != 0:
                # get the best iou for each db_bbox
                best_ious = np.max(ious, axis=1)
                best_ious_idx = np.argmax(ious, axis=1)
            # show the mask
            # plt.figure()
            fig, ax = plt.subplots(1,1, figsize=(5,5))
            ax.imshow(image,cmap='gray')
            ax.imshow(mask, alpha=0.5)
            # show the db_bboxes
            for i, bbox in enumerate(db_bboxes):
                linestyle = '-'
                if i in unmatched_db:
                    linestyle = '--'
                ax.add_patch(patches.Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],linewidth=1,edgecolor='b',facecolor='none', linestyle=linestyle))
                #format best iou to .2f
                b=best_ious[i]
                b = "{:.2f}".format(b)
                ax.text(bbox[0],bbox[1], b, color='b')
            # show the groups_bboxes
            for i, bbox in enumerate(groups_bboxes):
                color = 'g'
                if i in bad_ms:
                    color = 'r'
                elif i in unmatched_ms:
                    color = 'y'
                ax.add_patch(patches.Rectangle((bbox[0],bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],linewidth=1,edgecolor=color,facecolor='none'))
                if i in bad_ms:
                    ax.text(bbox[0],bbox[1], 'bad', color='r')

            for i, match in enumerate(matches):
                if match != -1:
                    ax.plot([db_bboxes[i][0], groups_bboxes[match][0]], [db_bboxes[i][1], groups_bboxes[match][1]], color='g')
            
            plt.show()

        return basename, out_groups, cur_out_stats





