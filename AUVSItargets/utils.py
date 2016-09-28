from __future__ import division
import math
import numpy as np
import functools
import hashlib
import random
import cv2

import AUVSItargets.global_settings as gs

__all__ = (
    'memoized',
    'array2lmdb',
    'loadFDdata',
    'squareCoords',
    'tightCrop',
    'angle2dcm'
)


def squareCoords(coords, noise=False):
    """Return coords in the form of a square, centered around the original
    coords. Possibly perturbation the coords."""

    patch_h, patch_w = coords[2]-coords[0], coords[3]-coords[1]
    patch_hw = max(patch_h, patch_w)

    if noise:
        tmp = int(patch_hw*random.uniform(0, gs.PATCH_COORDS_NOISE))
        patch_hw += tmp

    patch_cx = int((coords[2]+coords[0]-patch_hw)/2)
    patch_cy = int((coords[3]+coords[1]-patch_hw)/2)

    if noise:
        patch_cx += int(random.randint(-tmp, tmp)/2)
        patch_cy += int(random.randint(-tmp, tmp)/2)

    square_coords = max(patch_cx, 0), \
                    max(patch_cy, 0), \
                    patch_cx+patch_hw, \
                    patch_cy+patch_hw

    return square_coords


def tightCrop(mask):
    #
    # Calculate the boundries of the 'letter'
    #
    x_nnz = np.squeeze(mask).astype(np.float32).sum(axis=0).nonzero()[0]
    y_nnz = np.squeeze(mask).astype(np.float32).sum(axis=1).nonzero()[0]

    rect = np.array([x_nnz[0], y_nnz[0],
        x_nnz[-1]-x_nnz[0], y_nnz[-1]-y_nnz[0]])

    #
    # Square the rectangle so the the scaling will not distort the letter.
    #
    if rect[2] > rect[3]:
        d = int((rect[2]-rect[3])/2)
        rect[3] = rect[2]
        rect[1] = max(0, rect[1]-d)
    elif rect[3] > rect[2]:
        d = int((rect[3]-rect[2])/2)
        rect[2] = rect[3]
        rect[0] = max(0, rect[0]-d)

    #
    # Calculate an affine transform that centers the letter.
    #
    src = np.array(((rect[0], rect[1]), (rect[0], rect[1]+rect[3]),
        (rect[0] + rect[2], rect[1] + rect[3])), dtype=np.float32)
    LD_MARGIN = gs.LETTER_PATCH_SIZE[0] - gs.LETTER_MARGIN
    dst = np.array(
        (
            (gs.LETTER_MARGIN, gs.LETTER_MARGIN),
            (gs.LETTER_MARGIN, LD_MARGIN),
            (LD_MARGIN, LD_MARGIN)
            ),
        dtype=np.float32
    )
    M = cv2.getAffineTransform(src, dst)
    mask = cv2.warpAffine(
        mask,
        M,
        dsize=gs.LETTER_PATCH_SIZE,
        flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return mask


class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func, onlylast=False):
        self.__func = func
        self.__cache = {}
        self.__onlylast = onlylast

    def __call__(self, *args, **kwargs):
        ckey = hashlib.sha1(self.__func.__name__)
        for a in args:
            ckey.update(repr(a))
        for k in sorted(kwargs):
            ckey.update("%s:%s" % (k, repr(kwargs[k])))
        ckey = ckey.hexdigest()

        if ckey in self.__cache:
            result = self.__cache[ckey]
        else:
            result = self.__func(*args, **kwargs)
            if self.__onlylast:
                self.__cache = {}
            self.__cache[ckey] = result

        return result

    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


def bytesize(arr):
    """ Returns the memory byte size of a Numpy array as an integer."""

    byte_size = np.prod(arr.shape) * np.dtype(arr.dtype).itemsize

    return byte_size


def array2lmdb(X, y, lmdb_path):
    import caffe
    import lmdb

    N = X.shape[0]

    # We need to prepare the database for the size. If you don't have
    # deepdish installed, just set this to something comfortably big
    # (there is little drawback to settings this comfortably big).
    map_size = bytesize(X) * 2

    env = lmdb.open(lmdb_path, map_size=map_size)

    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tobytes()
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        with env.begin(write=True) as txn:
            # txn is a Transaction object
            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())


def loadFDdata(base_folder, test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """

    FTRAIN = os.path.join(base_folder, 'training.csv')
    FTEST = os.path.join(base_folder, 'test.csv')

    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


def angle2dcm(yaw, pitch, roll, input_units='rad', rotation_sequence='321'):
    """
    Returns a transformation matrix (aka direction cosine matrix or DCM) which
    transforms from navigation to body frame.  Other names commonly used,
    besides DCM, are `Cbody2nav` or `Rbody2nav`.  The rotation sequence
    specifies the order of rotations when going from navigation-frame to
    body-frame.  The default is '321' (i.e Yaw -> Pitch -> Roll).

    Parameters
    ----------
    yaw   : yaw angle, units of input_units.
    pitch : pitch angle, units of input_units.
    roll  : roll angle , units of input_units.
    input_units: units for input angles {'rad', 'deg'}, optional.
    rotationSequence: assumed rotation sequence {'321', others can be
                                                implemented in the future}.

    Returns
    -------
    Rnav2body: 3x3 transformation matrix (numpy matrix data type).  This can be
               used to convert from navigation-frame (e.g NED) to body frame.

    Notes
    -----
    Since Rnav2body is a proper transformation matrix, the inverse
    transformation is simply the transpose.  Hence, to go from body->nav,
    simply use: Rbody2nav = Rnav2body.T

    Reference
    ---------
    [1] Equation 2.4, Aided Navigation: GPS with High Rate Sensors,Jay A. Farrel 2008
    [2] eul2Cbn.m function (note, this function gives body->nav) at:
    http://www.gnssapplications.org/downloads/chapter7/Chapter7_GNSS_INS_Functions.tar.gz

    Copyright (c) 2014, NavPy Developers
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    # Apply necessary unit transformations.
    if input_units == 'rad':
        pass
    elif input_units == 'deg':
        yaw, pitch, roll = np.radians([yaw, pitch, roll])

    # Build transformation matrix Rnav2body.
    s_r, c_r = math.sin(roll) , math.cos(roll)
    s_p, c_p = math.sin(pitch), math.cos(pitch)
    s_y, c_y = math.sin(yaw)  , math.cos(yaw)

    if rotation_sequence == '321':
        # This is equivalent to Rnav2body = R(roll) * R(pitch) * R(yaw)
        # where R() is the single axis rotation matrix.  We implement
        # the expanded form for improved efficiency.
        Rnav2body = np.array([
                [c_y*c_p               ,  s_y*c_p              , -s_p    ],
                [-s_y*c_r + c_y*s_p*s_r,  c_y*c_r + s_y*s_p*s_r,  c_p*s_r],
                [ s_y*s_r + c_y*s_p*c_r, -c_y*s_r + s_y*s_p*c_r,  c_p*c_r]])

    else:
        # No other rotation sequence is currently implemented
        print('WARNING (angle2dcm): requested rotation_sequence is unavailable.')
        print('                     NaN returned.')
        Rnav2body = np.nan

    return Rnav2body


