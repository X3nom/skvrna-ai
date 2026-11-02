import time

from copy import deepcopy
import cv2
import numpy as np
from skimage.measure import label, regionprops, points_in_poly

from albumentations.core.transforms_interface import DualTransform, BasicTransform
from albumentations.augmentations.transforms import Flip

import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

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

class DeepsunClassificationRandomFlip(Flip):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    def __call__(self, *args, force_apply=False, **kwargs):
        mod_kwargs = deepcopy(kwargs)
        print("DeepsunClassificationRandomFlip: ", mod_kwargs.keys())
        print([type(x) for x in mod_kwargs.values()])
        print()
        mod_kwargs['image'] = kwargs['image']
        mod_kwargs['mask'] = kwargs['mask']
        processed_kwargs = super().__call__(*args, force_apply=force_apply, **mod_kwargs)

        kwargs['image'] = processed_kwargs['image']
        kwargs['mask'] = processed_kwargs['mask']
        return kwargs


class DeepsunRotateAndCropAroundGroup_Focus_Move(DualTransform):

    def __init__(self, standard_height, standard_width, 
                        focus_on_group=True,
                        random_move=False, random_move_percent=0.1,  
                        always_apply=False, p=1.0):     
        super().__init__(always_apply=always_apply, p=p)
        self.standard_height = standard_height
        self.standard_width = standard_width

        self.focus_on_group = focus_on_group
        self.random_move = random_move
        self.random_move_percent = random_move_percent

        self.index = 0

    def get_transform_init_args_names(self):
        return ("standard_height", "standard_width", "focus_on_group", "random_move", "random_move_percent")

    def expand_small_spots(self, msk):
        out_msk = msk.copy()
        label_img = label(out_msk)
        regions = regionprops(label_img)
        
        for r in regions:
            if r.area == 1:
                coords = r.coords[0]
                # print(coords)
                out_msk[coords[0]-1:coords[0]+1,coords[1]-1:coords[1]+1] = msk[coords[0],coords[1]]
                
        return out_msk

    # get mask contaoining only components at given coordinates using regionprops
    def get_mask_from_coords(self, mask, coords):
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

    def padder(self, vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    def get_bounding_box_around_group_with_padding(self, mask, offset):
        # Get the bounding box around non-zero pixels in mask
        x, y = np.nonzero(mask)
        x1, x2 = np.min(x), np.max(x)
        y1, y2 = np.min(y), np.max(y)

        # Add padding
        x1 -= offset
        x2 += offset
        y1 -= offset
        y2 += offset

        # Make sure the bounding box is not outside the image
        x1 = max(x1, 0)
        x2 = min(x2, mask.shape[0])
        y1 = max(y1, 0)
        y2 = min(y2, mask.shape[1])

        return x1, x2, y1, y2

    def adapt_bbox_to_image_size(self, bbox, image_size):
        bbox_center = ((bbox[0] + bbox[1]) // 2, (bbox[2] + bbox[3]) // 2)
        bbox_size = (bbox[1] - bbox[0], bbox[3] - bbox[2])

        # if bbox is too small, expand it
        minimal_percentage = .4

        bbox_size = (max(bbox_size[0], image_size[0] * minimal_percentage),
                     max(bbox_size[1], image_size[1] * minimal_percentage))
        
        return (int(bbox_center[0] - bbox_size[0] // 2), int(bbox_center[0] + bbox_size[0] // 2),
                int(bbox_center[1] - bbox_size[1] // 2), int(bbox_center[1] + bbox_size[1] // 2))
    
    def data_aug_random_move(self, bbox, max_offset):
        '''
        Randomly move the bounding box
        param bbox: bounding box
        param max_offset: maximum offset in portion of the bbox size
        '''
        # Randomly move the bounding box
        x1, x2, y1, y2 = bbox
        horizontal_offset = (np.random.random(1) * 2*max_offset) - max_offset
        vertical_offset = (np.random.random(1) * 2*max_offset) - max_offset
        # print(f'horizontal_offset: {horizontal_offset}, vertical_offset: {vertical_offset}')

        x1 += int(horizontal_offset * (bbox[1] - bbox[0]))
        x2 += int(horizontal_offset * (bbox[1] - bbox[0]))
        y1 += int(vertical_offset * (bbox[3] - bbox[2]))
        y2 += int(vertical_offset * (bbox[3] - bbox[2]))

        return x1, x2, y1, y2

    def padding(self, array, xx, yy):
        """
        :param array: numpy array
        :param xx: desired height
        :param yy: desirex width
        :return: padded array
        """

        h = array.shape[0]
        w = array.shape[1]

        a = (xx - h) // 2
        aa = xx - a - h

        b = (yy - w) // 2
        bb = yy - b - w

        return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

    def crop_img(self, img, bbox):
        # Crop image
        x1, x2, y1, y2 = bbox
        img = img[x1:x2, y1:y2]
        return img

    # crop image using bounding box and pad it to standard size
    def crop_and_pad(self, img, bbox, image_size):
        # Crop image
        img = self.crop_img(img, bbox)

        # Pad image
        img = self.padding(img, image_size[0], image_size[1])

        return img

    def __call__(self, *args, force_apply=False, **kwargs):
        # print('DeepsunRotateAndCropAroundGroup')

        st = time.time()

        img = kwargs['image']
        msk = kwargs['mask']
        disk = kwargs['solar_disk'].astype(np.uint8)

        # print(disk.dtype)

        try:
            # Make sure that all the single-pixels spots in mask are expanded to 3x3
            msk = self.expand_small_spots(msk)
            # 1) Correct solar Angle ->  Rotate image + Zoom In
            angle = kwargs['solar_angle']
            deltashapeX = kwargs['deltashapeX']
            deltashapeY = kwargs['deltashapeY']
            # print('solar_angle', angle)

            # rot_img = rotate(img, angle=angle, reshape=True)
            # rot_msk = rotate(msk, angle=angle, reshape=True)
            #rot_img = rotate_CV_bound(img, angle=angle, interpolation=cv2.INTER_LINEAR)
            #rot_msk = rotate_CV_bound(msk, angle=angle, interpolation=cv2.INTER_LINEAR)
            rot_img = rotate_CV_bound(img, angle=angle, interpolation=cv2.INTER_NEAREST)
            rot_msk = rotate_CV_bound(msk, angle=angle, interpolation=cv2.INTER_NEAREST)
            rot_disk = rotate_CV_bound(disk, angle=angle, interpolation=cv2.INTER_NEAREST)

            rot_img_zoom = rot_img[deltashapeX//2:rot_img.shape[0]-deltashapeX//2,
                            deltashapeY//2:rot_img.shape[1]-deltashapeY//2] 
            rot_msk_zoom = rot_msk[deltashapeX//2:rot_msk.shape[0]-deltashapeX//2,
                            deltashapeY//2:rot_msk.shape[1]-deltashapeY//2] 
            rot_disk_zoom = rot_disk[deltashapeX//2:rot_disk.shape[0]-deltashapeX//2,
                            deltashapeY//2:rot_disk.shape[1]-deltashapeY//2] 

            # print(rot_img_zoom.shape, rot_msk_zoom.shape)
            assert rot_img_zoom.shape == rot_msk_zoom.shape

            grp_mask = self.get_mask_from_coords(rot_msk_zoom, kwargs['members'])
            grp_rot_msk_zoom = rot_msk_zoom * grp_mask

            # 2) Crop around group
            # group_centroid = np.array(kwargs['centroid_px'])
            group_centroid = np.array((kwargs['members_mean_px'][1], kwargs['members_mean_px'][0]))

            # print(group_centroid)

            # minX = int(group_centroid[0])-self.standard_width//2
            # maxX = int(group_centroid[0])+self.standard_width//2
            # minY = int(group_centroid[1])-self.standard_height//2
            # maxY = int(group_centroid[1])+self.standard_height//2

            # img_group_crop = rot_img_zoom[minX:maxX,minY:maxY]
            # msk_group_crop = rot_msk_zoom[minX:maxX,minY:maxY]

            minX = self.standard_height + (int(group_centroid[1])-self.standard_width//2)
            maxX = self.standard_height + (int(group_centroid[1])+self.standard_width//2)
            minY = self.standard_height + (int(group_centroid[0])-self.standard_height//2)
            maxY = self.standard_height + (int(group_centroid[0])+self.standard_height//2)


            pad_rot_img_zoom = np.pad(rot_img_zoom, self.standard_height, self.padder, padder=0)
            pad_rot_msk_zoom = np.pad(rot_msk_zoom, self.standard_height, self.padder, padder=0)
            pad_grp_rot_msk_zoom = np.pad(grp_rot_msk_zoom, self.standard_height, self.padder, padder=0)
            pad_rot_disk_zoom = np.pad(rot_disk_zoom, self.standard_height, self.padder, padder=0)

            img_group_crop = pad_rot_img_zoom[minX:maxX,minY:maxY]
            msk_group_crop = pad_rot_msk_zoom[minX:maxX,minY:maxY]
            grp_msk_group_crop = pad_grp_rot_msk_zoom[minX:maxX,minY:maxY]
            disk_group_crop = pad_rot_disk_zoom[minX:maxX,minY:maxY]

            # Get the bounding box around the group
            bbox = self.get_bounding_box_around_group_with_padding(grp_msk_group_crop, 10)
            # print('members', kwargs['members'])
            # print('group_centroid', group_centroid,'bbox', bbox)
                
            # Crop the image and mask around the group while keeping the same size
            if self.focus_on_group:
                # focus on the group
                # Modify the bounding box if data augmentation is enabled
                if self.random_move:
                    bbox = self.data_aug_random_move(bbox, max_offset=self.random_move_percent)
                    
                    # Make sure the bounding box is not outside the image
                    x1, x2, y1, y2 = bbox
                    x1 = max(x1, 0)
                    x2 = min(x2, self.standard_width)
                    y1 = max(y1, 0)
                    y2 = min(y2, self.standard_height)
                    bbox = x1, x2, y1, y2

                bbox = self.adapt_bbox_to_image_size( bbox, (self.standard_height, self.standard_width))
                img_group_crop = self.crop_and_pad(img_group_crop, bbox, (self.standard_height, self.standard_width))
                msk_group_crop = self.crop_and_pad(msk_group_crop, bbox, (self.standard_height, self.standard_width))
                grp_msk_group_crop = self.crop_and_pad(grp_msk_group_crop, bbox, (self.standard_height, self.standard_width))
                disk_group_crop = self.crop_and_pad(disk_group_crop, bbox, (self.standard_height, self.standard_width))
            
            else:
                if self.random_move:
                    frac = np.max([(bbox[1]-bbox[0]) /self.standard_height, (bbox[3]-bbox[2]) /self.standard_width])
                    frac = np.sqrt(frac)
                    # print(minX, maxX, minY, maxY , frac, self.random_move_percent)
                    bbox = self.data_aug_random_move([minX,maxX,minY,maxY], max_offset=self.random_move_percent*frac)
                    minX,maxX,minY,maxY = bbox
                    # print(minX, maxX, minY, maxY )
                    
                    img_group_crop = pad_rot_img_zoom[minX:maxX,minY:maxY]
                    msk_group_crop = pad_rot_msk_zoom[minX:maxX,minY:maxY]
                    grp_msk_group_crop = pad_grp_rot_msk_zoom[minX:maxX,minY:maxY]
                    disk_group_crop = pad_rot_disk_zoom[minX:maxX,minY:maxY]


            

            assert img_group_crop.shape == msk_group_crop.shape

        except AssertionError:
            print(rot_img_zoom.shape,rot_msk_zoom.shape)
            raise
            pass
            # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
            # img = kwargs['image'].copy()
            # ax[0].set_title(kwargs["name"])
            # ax[1].set_title(kwargs["centroid_px"])
            # ax[0].imshow(img, interpolation=None, cmap='gray')
            # ax[0].imshow(msk, interpolation=None, alpha=0.5)
            # ax[1].imshow(rot_img, interpolation=None, cmap='gray')
            # ax[1].imshow(rot_msk, interpolation=None, alpha=0.5)
            # ax[2].imshow(img_group_crop, interpolation=None, cmap='gray')
            # ax[2].imshow(msk_group_crop, interpolation=None, alpha=0.5)
            
            # plt.show()
        
        # print(msk_group_crop.shape)

        kwargs.pop('solar_angle',None)
        kwargs.pop('deltashapeX',None)
        kwargs.pop('deltashapeY',None)
        # kwargs.pop('centroid_px',None)

        # plt.figure()
        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3))
        # ax[0].imshow(img, interpolation=None, cmap='gray')
        # ax[0].imshow(msk, interpolation=None, alpha=0.5)
        # ax[1].imshow(rot_img_zoom, interpolation=None, cmap='gray')
        # ax[1].imshow(rot_msk_zoom, interpolation=None, alpha=0.5)
        # ax[2].imshow(img_group_crop, interpolation=None, cmap='gray')
        # ax[2].imshow(msk_group_crop, interpolation=None, alpha=0.5)
        # ax[1].scatter(group_centroid[0],group_centroid[1], c='r', s=1 )
        # print("salut")
        # plt.show()
        # plt.savefig(f'./test_classification_{self.index}.png', dpi=150)
        
        self.index+=1


        kwargs['image'] = img_group_crop.copy()
        kwargs['mask'] = msk_group_crop.copy()
        kwargs['group_mask'] = grp_msk_group_crop.copy()
        kwargs['solar_disk'] = disk_group_crop.copy()


        et = time.time()
        # print(f'DeepsunRotateAndCropAroundGroup time: {et-st} seconds')
        
        
        return kwargs


# create a dual transform that selects only the sunpots in group.
# Implement __call__ and __init__ methods
class DeepsunSelectSunspotsInGroup(DualTransform):
    """Select only the sunspots in group. 
    """
    def __init__(self, always_apply=False, p=1.0):
        super(DeepsunSelectSunspotsInGroup, self).__init__(always_apply, p)

    def __call__(self, **kwargs):
        """Select only the sunspots in group. 
        """
        st = time.time()

        img = kwargs['image']
        msk = kwargs['mask']




