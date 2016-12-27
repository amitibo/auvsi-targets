from __future__ import division
import exifread
import numpy as np
from datetime import datetime
import traceback
import random
import math
import cv2
import json
import os

import AUVSItargets.global_settings as gs
import AUVSItargets.NED
from AUVSItargets.utils import angle2dcm

__all__ = [
    "Image",
]

in_to_mm = 25.4


def tagRatio(tag):
    ratio = tag.values[0].num/tag.values[0].den
    return ratio


def tagValue(tag):
    return tag.values[0]


def overlay(img, target, M, intensity_scale,
    center_patch=False, bounding_rect=False):

    #
    # Calculate the destination pixels of the patch. This allows for much more
    # efficient copies (instead of copying a full 6000x4000 image).
    #
    offsets, dst_shape, shifts = calcDstLimits(img, target, M, center_patch)

    if dst_shape[0] == 0 or dst_shape[1] == 0:
        #
        # Targets outside of the frame are not pasted.
        #
        return

    T = np.eye(3)
    T[0, 2] = -offsets[0]
    T[1, 2] = -offsets[1]

    if center_patch:
        T1 = np.eye(3)
        T1[0, 2] = -shifts[0]
        T1[1, 2] = -shifts[1]

        T = np.dot(T, T1)

    #
    # Draw the target template
    #
    target.drawTemplate(dst_shape, np.dot(T, M))
    overlay_img, overlay_alpha = target.img, target.alpha

    #
    # Scale the intensity and add poisson noise to the overlay.
    #
    overlay_yuv = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2YUV).astype(np.float)
    overlay_yuv[:, :, 0] *= intensity_scale
    overlay_yuv[:, :, 0] = np.random.poisson(
        overlay_yuv[:, :, 0]*gs.POISSON_INTENSITY_RATIO, overlay_img.shape[:2]
        )/gs.POISSON_INTENSITY_RATIO
    overlay_yuv[overlay_yuv > 255] = 255
    overlay_img = cv2.cvtColor(overlay_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)

    #
    # Paste the img crop as background to the overlay so that the blur will
    # not create artifacts.
    #
    overlay_alpha = overlay_alpha[..., np.newaxis]
    overlay_img = (
        img[offsets[1]:offsets[1]+dst_shape[1],
        offsets[0]:offsets[0]+dst_shape[0], :3].astype(np.float32)*
        (1-overlay_alpha) +
        overlay_img[..., :3].astype(np.float32)*overlay_alpha
        ).astype(np.uint8)

    #
    # Smoothen the overlay
    # Note:
    # There is no need to smoothen very small targets as this happens in
    # scale down operation.
    #
    if overlay_img.shape[0] > 10:
        ksize = 3
        ksigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
        overlay_img = cv2.GaussianBlur(overlay_img, (ksize, ksize), ksigma)

    #
    # Blend the image and overlay.
    #
    img[offsets[1]:offsets[1]+dst_shape[1],
        offsets[0]:offsets[0]+dst_shape[0],
        :3] = (img[
                    offsets[1]:offsets[1]+dst_shape[1],
                    offsets[0]:offsets[0]+dst_shape[0],
                    :3].astype(np.float32)*(1-overlay_alpha) +
               overlay_img[..., :3].astype(np.float32)*overlay_alpha
            ).astype(np.uint8)

    if not bounding_rect:
        return None

    #
    # Calculate a tight bounding rect
    #
    binary_img = (np.squeeze(overlay_alpha) > 0).astype(np.uint8)
    contour = cv2.findContours(binary_img, mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE)
    if cv2.__version__[0] == '3':
        #
        # It seems that in version 3.1.0 (and possibly other >3 versions) the
        # findContours function returns: binary_img, contours, hierarchy tuple.
        #
        rect = cv2.boundingRect(points=contour[1][0])        
    else:
        #
        # In opencv version 2.4 the output is the countours.
        #
        rect = cv2.boundingRect(points=contour[0][0])

    return rect[0]+offsets[0], \
           rect[1]+offsets[1], \
           rect[0]+offsets[0]+rect[2], \
           rect[1]+offsets[1]+rect[3]


def calcDstLimits(img, target, M, center_patch):
    """Calculate the limits of the overlay in the destination image.
    """

    isize = img.shape[:2]

    limits = np.float32((((0, 0), (0, target.size),
        (target.size, target.size), (target.size, 0)),))

    limits_trans = cv2.perspectiveTransform(limits, M)
    dst_xlimit = cv2.minMaxLoc(limits_trans[0, :, 0])[:2]
    dst_ylimit = cv2.minMaxLoc(limits_trans[0, :, 1])[:2]

    #
    # Center the patch
    #
    if center_patch:
        x_shift = (dst_xlimit[1]+dst_xlimit[0] - isize[1])/2
        y_shift = (dst_ylimit[1]+dst_ylimit[0] - isize[0])/2

        dst_xlimit = [min(max(int(dst_xlimit[0]-x_shift), 0), isize[1]),
            min(max(int(dst_xlimit[1]-x_shift+1), 0), isize[1])]
        dst_ylimit = [min(max(int(dst_ylimit[0]-y_shift), 0), isize[0]),
            min(max(int(dst_ylimit[1]-y_shift+1), 0), isize[0])]
    else:
        x_shift, y_shift = 0, 0
        dst_xlimit = [min(max(int(dst_xlimit[0]), 0), img.shape[1]),
            min(max(int(dst_xlimit[1]+1), 0), img.shape[1])]
        dst_ylimit = [min(max(int(dst_ylimit[0]), 0), img.shape[0]),
            min(max(int(dst_ylimit[1]+1), 0), img.shape[0])]

    offsets = (dst_xlimit[0], dst_ylimit[0])
    shape = (dst_xlimit[1]-dst_xlimit[0], dst_ylimit[1]-dst_ylimit[0])

    return offsets, shape, (x_shift, y_shift)


class Image(object):
    """The Image class

    This Image class is used for encapsulating both the image and its telemetry
    data. It used both on the airborne system when capturing images
    (with time stamp) and on the ground system where it used for loading images
    and their respective flight data and manipulating those in the GUI.
    """

    def __init__(self, img_path=None, data_path=None, timestamp=None,
        intensity=gs.EMPIRICAL_IMAGE_INTENSITY,
        img_path_full_size=None, K=None):

        self._stitching_keypoints = None
        self._stitching_destination = None
        self._tags = None
        self._intensity = intensity
        self._path = img_path
        self._img_path_full_size = img_path_full_size

        #
        # Load image
        #
        if img_path is not None:
            self._img = cv2.imread(img_path)

            if self._img is None:
                raise Exception(
                    'Could not load image {img}'.format(img=img_path)
                    )

            #
            # Some 'preprocessing'
            # This is used for calculating Quads in the map widget. It needs
            # to use the original image dimensions used for calculating K matrix
            #
            h, w, _ = self._img.shape

            self._limits = np.array(
                (
                    (0, w, w, 0),
                    (0, 0, h, h),
                    (1, 1, 1, 1.)
                )
            )

        else:
            self._img = None

        #
        # Load flight data
        #
        self._K = None
        self._Kinv = None
        if data_path is not None:
            try:
                with open(data_path, 'rb') as f:
                    self._flight_data = json.load(f)

                if K is not None:
                    self._K = K
                elif 'K' in self._flight_data.keys():
                    self._K = np.array(self._flight_data['K'])
                else:
                    print 'No intrinsic matrix given.'

                if self._K is not None:
                    self._Kinv = np.linalg.inv(self._K)

                self._datetime = self._flight_data['timestamp']

                #
                # Calculate extrinsic Matrix.
                #
                if 'yaw' in self._flight_data and \
                   self._flight_data['src_att'] is not None:
                    #
                    # The yaw is taken from the Pixhawk.
                    # The reason is the we don't know to use the vector nav
                    # properly.
                    #
                    if self._flight_data['src_att'] == u'PixHawk':
                        yaw = math.degrees(self._flight_data['yaw'])
                    else:
                        yaw = math.degrees(
                            self._flight_data['all']['PixHawk']['yaw']
                            )
                elif 'cog' in self._flight_data and \
                     self._flight_data['src_cog'] is not None:
                    yaw = self._flight_data['cog'] / 100

                else:
                    yaw = 0

                if 'pitch' in self._flight_data and \
                   self._flight_data['src_att'] is not None:
                    pitch = math.degrees(self._flight_data['pitch'])
                else:
                    pitch = 0

                if 'roll' in self._flight_data and \
                   self._flight_data['src_att'] is not None:
                    roll = math.degrees(self._flight_data['roll'])
                else:
                    roll = 0

                self._stitching_pitch = pitch
                self._stitching_roll = roll
                self._stitching_yaw = yaw
                self.stitching_pic_OK = 1

                #
                # Calculate the extrinsic matrix.
                #
                self.calculateExtrinsicMatrix(
                    latitude=self._flight_data['lat']*1e-7,
                    longitude=self._flight_data['lon']*1e-7,
                    altitude=self._flight_data['relative_alt']*1e-3,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                )

                #
                # Plane
                #
                self.plane = {
                    'pitch': self._flight_data['pitch'],
                    'roll': self._flight_data['roll']
                }
            except:
                print traceback.format_exc()
                data_path = None

        if data_path is None:
            #
            # Get the time stamp.
            #
            if timestamp is not None:
                self._datetime = timestamp
            elif 'Image DateTime' in self.tags:
                img_dt = self.tags['Image DateTime'].values
                img_dt = img_dt.replace(':', '_').replace(' ', '_')
                current_dt = datetime.now().strftime("_%f")
                self._datetime = img_dt + current_dt
            else:
                print 'No Image DateTime tag. Using computer time.'
                self._datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    def stitching_detectFeatures(self, featureDetector):
        if featureDetector is None:
            raise Exception('image: stitching_detectFeatures - exception: \
                feature detector is None')
        #if not isinstance(featureDetector, Area):
        #    raise TypeError("area must be set to an Area")
        full_image = cv2.imread(self._img_path_full_size)

        self._stitching_keypoints, self._stitching_destination = \
            featureDetector.detectAndCompute(full_image, None)

    def stitching_getImageFeatures(self):
        return self._stitching_keypoints, self._stitching_destination

    def calculateExtrinsicMatrix(self, latitude, longitude,
        altitude, yaw, pitch, roll):
        """Calculate camera extrinsic matrix

        Calculate the extrinsic matrix in local Cartesian mapping (NED) which
        is centered at the camera.
        """

        #
        # Calculate camera extrinsic matrix
        # Note:
        # The local Cartesian mapping (NED) is centered at the camera (therefore
        # the translation matrix is an eye matrix).
        #
        self._Rt = np.eye(4)
        self._Rt[:3, :3] = angle2dcm(yaw, pitch, roll, input_units='deg').T

        self._latitude = latitude
        self._longitude = longitude
        self._altitude = altitude
        self._yaw = yaw
        self._pitch = pitch
        self._roll = roll

    def paste(self, target, intensity_scale=1):
        """Draw a target on the image.

        This function uses the parameters of the target and the image to
        calculate the location and then draw the target on the image.

        Parameters
        ----------
        target : Target object.
            Target to draw on the image, should be an object from a subclass of
            AUVSItargets.BaseTarget.
        """

        #
        # Calculate the transform matrix from the target coordinates to the
        # camera coordinates.
        #
        target_H = target.H(
            latitude=self._latitude,
            longitude=self._longitude,
            altitude=self._altitude
        )
        M1 = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 1)))
        M2 = np.array(((0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 1, 0)))
        M = np.dot(
            self.K,
            np.dot(M2, np.dot(np.linalg.inv(self.Rt), np.dot(target_H, M1)))
            )

        overlay(img=self._img, target=target,
            M=M, intensity_scale=intensity_scale)

    def createPatches(self, patch_size, patch_shift):
        """Create patches(crops) of an image

        This function crops patches of an image on a regularly spaced grid.

        Parameters
        ----------
        patch_size : int
            Scalar size (both width and height) of a rectangular patch.
        patch_shift : int
            Space (both horizontal and vertical) between patches.
        """

        patch_height, patch_width = patch_size
        nx = int((self._img.shape[1] - patch_width)/patch_shift)
        ny = int((self._img.shape[0] - patch_height)/patch_shift)

        for i in range(nx):
            for j in range(ny):
                sx = i*patch_shift
                sy = j*patch_shift
                patch = self._img[sy:sy+patch_height, sx:sx+patch_width, :]
                yield patch.copy()

    def createRandomSizedPatches(self, patch_size_range, patch_shift):
        """Create patches(crops) of an image

        This function crops patches of random size of an image on a regularly
        spaced grid.

        Parameters
        ----------
        patch_size_range : tuple of ints
            range of sizes for a square patch.
        patch_shift : int
            Space (both horizontal and vertical) between patches.
        """

        patch_min, patch_max = patch_size_range
        nx = int((self._img.shape[1] - patch_max)/patch_shift)
        ny = int((self._img.shape[0] - patch_max)/patch_shift)

        for i in range(nx):
            for j in range(ny):
                sx = i*patch_shift
                sy = j*patch_shift
                patch_size = random.randint(patch_min, patch_max)
                patch = self._img[sy:sy+patch_size, sx:sx+patch_size, :]
                yield patch.copy()

    def pastePatch(self, patch, target, intensity_scale=1):
        """Paste a target on a patch

        The target is pasted in the center of the patch (the coords of the
        patch and target are ignored).

        Parameters
        ----------
        patch: array
            The patch on which the target is pasted into.
        target: target object.
            The target to paste on the patch.
        """

        target_H = target.H(
            latitude=self._latitude,
            longitude=self._longitude,
            altitude=self._altitude
        )
        M1 = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 1)))
        M2 = np.array(((0, 1, 0, 0), (-1, 0, 0, 0), (0, 0, 1, 0)))
        M = np.dot(self.K,
            np.dot(M2, np.dot(np.linalg.inv(self.Rt), np.dot(target_H, M1))))

        bounding_rect = overlay(img=patch, target=target, M=M,
            intensity_scale=intensity_scale, center_patch=True,
            bounding_rect=True)
        return bounding_rect

    def calculateQuad(self, ned):

        x, y, h = ned.geodetic2ned([self._latitude, self._longitude,
            self._altitude])

        offset = np.array(
            (
                (x,),
                (y,),
            )
        )

        #
        # Project the image corners to camera axes (at z=1)
        #
        p = np.dot(self._Kinv, self._limits)

        #
        # Switch x, y to convert from camera to local body coords.
        #
        p = p[(1, 0, 2), :]

        #
        # Rotate according the camera attitude (as measured by the VectorNav).
        # The directions are scaled so that their z coordinate will be equal
        # to 1.
        #
        r = np.dot(self.Rt[:3, :3], p)
        r = r[:2, ...] / r[2, ...].reshape(1, -1)

        #
        # project the corner directions to the ground
        #
        projections = offset + (-h) * r

        return projections[(1, 0), :]

    def coords2LatLon(self, px, py):
        """Convert image (flipped) coords to lat lon.

        Important Note
        --------------
        The coords assume that the 0,0 is at the bottom left of the image
        and therefore it is called flipped. This means that you need to
        flip the y coord when using a crop coords.
        """
        point = np.array(
            (
                (px,),
                (py,),
                (1,)
            )
        )


        #
        # Project the point to camera axes (at z=1)
        #
        p = np.dot(self._Kinv, point)

        #
        # Switch x, y to convert from camera to local body coords.
        #
        p = p[(1, 0, 2), :]


        #
        # Rotate according the camera attitude (as measured by the VectorNav).
        # The directions are scaled so that their z coordinate will be equal
        # to 1.
        #
        r = np.dot(self.Rt[:3, :3], p)

        r = r / r[2]


        #
        # project the point to the ground
        #
        ned = NED.NED(self._latitude, self._longitude, 0)
        x, y, h = ned.geodetic2ned([self._latitude, self._longitude,
            self._altitude])


        offset = np.array(
            (
                (x,),
                (y,),
                (h,)
            )
        )

        ned_coords = (offset + (-h) * r).flatten()

        lat, lon, alt = ned.ned2geodetic(ned=(ned_coords[0], ned_coords[1],
            ned_coords[2]))


        return lat, lon

    @property
    def img(self):
        return self._img

    @property
    def Rt(self):
        """Get the transform matrix of the camera.

        Returns the extrinsic parameters of the camera.
        """

        return self._Rt

    @property
    def K(self):
        """Get the K matrix of the camera.

        Retruns the intrinsic parameters of the camera.
        """

        return self._K

    @property
    def datetime(self):
        """Get date time tag"""

        return self._datetime

    @property
    def path(self):
        """Get path of image"""

        return self._path

    @property
    def name(self):
        """Get path of image"""

        return os.path.split(self._path)[-1]

    @property
    def latitude(self):

        return self._latitude

    @property
    def longitude(self):

        return self._longitude

    @property
    def altitude(self):

        return self._altitude

    @property
    def yaw(self):

        return self._yaw

    @property
    def shape(self):

        return self._img.shape

    @property
    def intensity(self):

        if self._intensity is None:
            self._intensity = np.mean(cv2.cvtColor(self._img,
                cv2.COLOR_BGR2GRAY))

        return self._intensity

    @property
    def tags(self):

        if self._tags is None:
            #
            # Get the EXIF data
            #
            with open(self.path, 'rb') as f:
                self._tags = exifread.process_file(f)

        return self._tags
