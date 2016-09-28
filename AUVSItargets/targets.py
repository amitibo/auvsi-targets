from __future__ import division
import numpy as np
import aggdraw
from PIL import Image
import random
import bisect
import string
import pickle
import cv2
import math
import tempfile
import shutil
import os

import AUVSItargets.global_settings as gs
import AUVSItargets.transformation_matrices as transforms
from AUVSItargets.NED import NED
from AUVSItargets.utils import angle2dcm

__all__ = [
    "CircleTarget",
    "HalfCircleTarget",
    "QuarterCircleTarget",
    "RectangleTarget",
    "TrapezoidTarget",
    "TriangleTarget",
    "CrossTarget",
    "PolygonTarget",
    "PentagonTarget",
    "HexagonTarget",
    "HeptagonTarget",
    "OctagonTarget",
    "StarTarget",
    "QRTarget",
    "randomTarget",
    "drawLetter",
    "randomLetter"
]


class WeightedRandomGenerator(object):
    def __init__(self, weights):
        self.totals = []
        running_total = 0

        for w in weights:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        rnd = random.random() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)

    def __call__(self):
        return self.next()


class BaseTarget(object):
    """
    Base target from which all other target inherit.

    size: float
        Size of target in meters.
    orientation: float
        Orientation of the target (degrees) in respect to the north.
    altitude: float
        Altitude of the target in meters
    longitude: float
        Longitude coord of the target
    latitude: float
        Latitude coord of the target
    color: three tuple
        Color of the target
    letter: char
        Letter to be drawn in the center of the target
    font_color: three tuple
        Color of the letter
    font_size: float
        Size of the letter, ratio to full target size.
    font: string
        Path of font to use
    template_size: int, optional(=400)
        Size of base template target (before pasting into drone image)
    """

    _text_offset_ratio = 1/2

    def __init__(
        self,
        size,
        orientation,
        altitude,
        longitude,
        latitude,
        color=None,
        letter=None,
        font_color=None,
        font=None,
        size_limits=gs.NORMAL_TARGET_SIZE_RANGE,
        ):

        if size is None:
            size_limits_diff = size_limits[1]-size_limits[0]
            self._size = random.random()*size_limits_diff + size_limits[0]
        else:
            self._size = size

        self._size_ratio = 100
        self._size *= self._size_ratio

        self._orientation = math.radians(orientation)
        self._altitude = altitude
        self._longitude = longitude
        self._latitude = latitude
        self._color = color
        self._letter = letter
        self._font_color = font_color
        self._font_size = self._size_ratio
        if font is None:
            self._font = random.choice(gs.FONTS)
        else:
            self._font = font

    def _drawForm(self, ctx, brush):
        """Draw the form of the target"""

        raise NotImplementedError()

    def _drawLetter(self, ctx):
        """Draw the letter on the target.

        This function is called after the target has been drawn.

        Parameters
        ----------
        ctx: aggdraw context
            The context to drawon.
        """

        font = aggdraw.Font(self._font_color, self._font, self._font_size)
        text_size = ctx.textsize(self._letter, font)
        position = [-text_size[i]/2 for i in range(2)]
        ctx.text(position, self._letter, font)

    def drawTemplate(self, target_shape, M):
        """Draw the target on the base template"""

        #
        # Prepare the template canvas
        #
        img = Image.new(
            mode='RGBA',
            size=target_shape,
            color=(255, 255, 255, 0)
        )
        ctx = aggdraw.Draw(img)
        brush = aggdraw.Brush(self._color, 255)

        #
        # Set the transform
        # Note:
        # aggdraw supports only affine transforms, so we use only the first 6
        # parameters of the projection transform.
        #
        C = np.array(
            (
                (1/self._size_ratio, 0, 0),
                (0, 1/self._size_ratio, 0),
                (0, 0, 1)
            )
        )
        M = np.dot(M, C)
        M = M/M[2, 2]
        ctx.settransform(M.ravel()[:6])

        #
        # Draw the form of the target
        #
        self._drawForm(ctx, brush)

        #
        # Add letter.
        #
        if self._letter is not None:
            #
            # The font is half the size of the form
            #
            C = np.array(
                (
                    (self._size/self._font_size/2, 0, self._size/2),
                    (0,
                     self._size/self._font_size/2,
                     self._size*self._text_offset_ratio
                    ),
                    (0, 0, 1)
                )
            )
            M = np.dot(M, C)
            ctx.settransform(M.ravel()[:6])

            self._drawLetter(ctx)
        #
        # Flush to apply drawing.
        #
        ctx.flush()

        img = np.array(img)
        self._templateImg = img[..., :3]
        self._templateAlpha = img[..., 3].astype(np.float32)/255

    def H(self, latitude, longitude, altitude):
        """Calculate the transform of the target.

        Calculate the Cartesian coordinate transform relative to a given
        latitude, longitude, alt coordinates.

        Parameters
        ----------
        latitude, longitude, altitude: three tuple of floats
            Center of the Cartesian coordinate system (e.g. camera center). The
            transform uses the local Cartesian coordinate system (North East
            Down).
            Makes use of code from the COLA2 project
            (https://bitbucket.org/udg_cirs/cola2).
        """

        #
        # Center the target around rotation center (center of weight).
        #
        T1 = transforms.translation_matrix((-self.size/2, -self.size/2, 0))

        #
        # Rotation.
        # Note: We subtract 90 degrees to convert between target coordinates
        # and world (NED) coordinates
        #
        R = np.eye(4)
        R[:3, :3] = angle2dcm(-self._orientation-math.pi/2, 0, 0,
            input_units='rad')

        #
        # Translation relative to center of axes (NED).
        #
        ned = NED(lat=latitude, lon=longitude, height=altitude)
        x, y, h = ned.geodetic2ned((self._latitude,
            self._longitude, self._altitude))
        T2 = transforms.translation_matrix((x, y, h))

        return np.dot(T2, np.dot(R, T1))

    @property
    def img(self):
        return self._templateImg

    @property
    def alpha(self):
        return self._templateAlpha

    @property
    def size(self):
        return self._size/self._size_ratio


class CircleTarget(BaseTarget):
    """A target in the form of a circle."""

    def _drawForm(self, ctx, brush):

        ctx.ellipse((0, 0, self._size, self._size), brush)


class HalfCircleTarget(BaseTarget):
    """A target in the form of a circle."""

    def _drawForm(self, ctx, brush):

        ctx.arc(
            (0, self._size/4, self._size, self._size+self._size/4),
            0,
            180,
            brush
        )


class QuarterCircleTarget(BaseTarget):
    """A target in the form of a circle."""

    def _drawForm(self, ctx, brush):

        offsetx = self._size/4
        offsety = self._size/5
        ctx.pieslice(
            (
                -offsetx,
                offsety,
                self._size+offsetx,
                self._size+2*offsetx+offsety
            ),
            45,
            135,
            brush
        )


class RectangleTarget(BaseTarget):
    """A target in the form of a rectangle."""

    def _drawForm(self, ctx, brush):

        height = self._size*random.uniform(0, 0.25)
        ctx.rectangle((0, height, self._size, self._size-height), brush)


class TrapezoidTarget(BaseTarget):
    """A target in the form of a rectangle."""

    def _drawForm(self, ctx, brush):

        offsetx = self._size*random.uniform(0.15, 0.25)
        offsety = self._size*random.uniform(0.15, 0.25)
        polygon = [
            offsetx, offsety,
            self._size-offsetx, offsety,
            self._size, self._size-offsety,
            0, self._size-offsety
        ]

        ctx.polygon(polygon, brush)


class TriangleTarget(BaseTarget):
    """A target in the form of a triangle."""

    def _drawForm(self, ctx, brush):

        self._text_offset_ratio = 2/3
        ctx.polygon((0, self._size, self._size, self._size, self._size/2, 0),
            brush)


class CrossTarget(BaseTarget):
    """A target in the form of a cross."""

    def _drawForm(self, ctx, brush):

        ctx.polygon(
            (
                0, self._size/3,
                0, self._size*2/3,
                self._size/3, self._size*2/3,
                self._size/3, self._size,
                self._size*2/3, self._size,
                self._size*2/3, self._size*2/3,
                self._size, self._size*2/3,
                self._size, self._size/3,
                self._size*2/3, self._size/3,
                self._size*2/3, 0,
                self._size/3, 0,
                self._size/3, self._size/3
                ),
            brush
        )


class PolygonTarget(BaseTarget):
    """A target in the form of a n-sided polygon."""

    def __init__(self, n=None, *args, **kwds):

        if n is None:
            n = random.randint(5, 8)
        self._nsides = n

        super(PolygonTarget, self).__init__(*args, **kwds)

    def _drawForm(self, ctx, brush):

        r = self._size/2
        alpha = np.pi*2/self._nsides

        polygon = []
        for i in range(self._nsides):
            polygon.append(r + r*np.cos(alpha*i))
            polygon.append(r + r*np.sin(alpha*i))

        ctx.polygon(polygon, brush)


class PentagonTarget(PolygonTarget):
    """A target in the form of a 5-sided polygon."""

    def __init__(self, *args, **kwds):
        super(PentagonTarget, self).__init__(n=5, *args, **kwds)


class HexagonTarget(PolygonTarget):
    """A target in the form of a 6-sided polygon."""

    def __init__(self, *args, **kwds):
        super(HexagonTarget, self).__init__(n=6, *args, **kwds)


class HeptagonTarget(PolygonTarget):
    """A target in the form of a 7-sided polygon."""

    def __init__(self, *args, **kwds):
        super(HeptagonTarget, self).__init__(n=7, *args, **kwds)


class OctagonTarget(PolygonTarget):
    """A target in the form of a 8-sided polygon."""

    def __init__(self, *args, **kwds):
        super(OctagonTarget, self).__init__(n=8, *args, **kwds)


class StarTarget(BaseTarget):
    """A target in the form of a n-star."""

    def __init__(self, n=None, *args, **kwds):

        if n is None:
            n = random.randint(5, 6)
        self._nstar = n

        super(StarTarget, self).__init__(*args, **kwds)

    def _drawForm(self, ctx, brush):

        r_outer = c = self._size/2
        r_inner = self._size/4
        alpha = np.pi/self._nstar

        polygon = []
        for i in range(2*self._nstar):
            if i % 2 == 1:
                r = r_outer
            else:
                r = r_inner

            polygon.append(c + r*np.cos(alpha*i))
            polygon.append(c + r*np.sin(alpha*i))

        ctx.polygon(polygon, brush)


class QRTarget(BaseTarget):
    """A target in the form of a circle."""

    def __init__(self, text=None,
        size_limits=gs.QR_TARGET_SIZE_RANGE,
        *args,
        **kwds):

        if text is None:
            text = 'www.{random_string}.com'.format(
                random_string=''.join(
                    random.choice(string.letters+string.digits+'_'
                    ) for _ in range(20))
            )
        self._text = text

        super(QRTarget, self).__init__(*args, size_limits=size_limits, **kwds)


    def drawTemplate(self, target_shape, M):
        """Draw the target on the base template"""

        import pyqrcode

        #
        # Create the QR code
        #
        qr = pyqrcode.create(self._text)
        base_path = tempfile.mkdtemp()
        img_path = os.path.join(base_path, 'temp.png')
        qr.png(img_path, scale=5)
        overlay_img = cv2.imread(img_path)
        shutil.rmtree(base_path)
        overlay_alpha = np.ones(overlay_img.shape[:2], dtype=np.float32)

        #
        # Paste the QR code onthe target shape.
        #
        C = np.array(
            (
                (self.size/overlay_img.shape[0], 0, 0),
                (0, self.size/overlay_img.shape[0], 0),
                (0, 0, 1)
            )
        )
        M = np.dot(M, C)
        flags = cv2.cv.CV_INTER_LINEAR+cv2.cv.CV_WARP_FILL_OUTLIERS
        self._templateImg = cv2.warpPerspective(overlay_img,
            M,
            dsize=target_shape,
            flags=flags)
        self._templateAlpha = cv2.warpPerspective(overlay_alpha,
            M,
            dsize=target_shape,
            flags=flags)


#
# Setup data for random selection.
# NOTE:
# Should match the list of shapes:
#SHAPE_LABELS = (
    #'Circle',
    #'Half Circle',
    #'Quarter Circle',
    #'Rectangle',
    #'Trapezoid',
    #'Triangle',
    #'Cross',
    #'Pentagon',
    #'Hexagon',
    #'Heptagon',
    #'Octagon',
    #'Star',
    #'QRcode',
    #'no target'
#)
TARGET_CLASSES = (
    (CircleTarget, 1., {}),
    (HalfCircleTarget, 1., {}),
    (QuarterCircleTarget, 1., {}),
    (RectangleTarget, 1., {}),
    (TrapezoidTarget, 1., {}),
    (TriangleTarget, 1., {}),
    (CrossTarget, 1., {}),
    (PentagonTarget, 1., {}),
    (HexagonTarget, 1., {}),
    (HeptagonTarget, 1., {}),
    (OctagonTarget, 1., {}),
    (StarTarget, 1., {}),
    (QRTarget, 1., {}),
)


with open(os.path.join(gs.DATA_PATH, 'colors.pkl'), 'r') as f:
    RGB_COLORS = pickle.load(f)

WRG = WeightedRandomGenerator(weights=zip(*TARGET_CLASSES)[1])
WRG_no_QRcode = WeightedRandomGenerator(weights=zip(*TARGET_CLASSES[:-1])[1])

def randomColor(ignore=None):

    color_list = RGB_COLORS.values()

    if ignore is not None:
        color_list.remove(ignore)

    return random.choice(color_list)


def randomTarget(longitude, latitude, altitude, target_label=None,
    coords_offset=0.0002, no_QRcode=False, **kwds):
    """Create a random target

    The target is selected randomly from all possible targets, and placed in a
    random offset from some given position.
    """

    color = randomColor()
    letter = random.choice(gs.LETTERS)
    params = {
        'size': None,
        'orientation': random.random()*360,
        'longitude': longitude + 2*(random.random()-0.5)*coords_offset,
        'latitude': latitude+2*(random.random()-0.5)*coords_offset,
        'altitude': altitude,
        'letter': letter,
        'color': color,
        'font_color': randomColor(ignore=color),
    }
    params.update(kwds)
    if target_label is None:
        if no_QRcode:
            target_label = WRG_no_QRcode.next()
        else:
            target_label = WRG.next()
    target, _, extra_params = TARGET_CLASSES[target_label]
    params.update(extra_params)

    return target(**params), target_label, gs.LETTERS.index(letter)


def drawLetter(letter, size=28, font=None, font_size=15):
    """Draw a mask of a letter (used for training a classifier)."""

    #
    # Prepare the template canvas
    #
    img = Image.new(
        mode='L',
        size=(size, size)
    )

    ctx = aggdraw.Draw(img)

    if font is None:
        import platform
        if platform.system() == 'Linux':
            font = r"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        else:
            font = r"C:\Windows\Fonts\Arialbd.ttf"

    C = np.array(
        (
            (size/font_size*4/5, 0, size/2),
            (0, size/font_size*4/5, size/2),
            (0, 0, 1)
        )
    )
    ctx.settransform(C.ravel()[:6])

    font = aggdraw.Font(255, font, font_size)
    text_size = ctx.textsize(letter, font)
    position = [-text_size[i]/2 for i in range(2)]
    ctx.text(position, letter, font)

    #
    # Flush to apply drawing.
    #
    ctx.flush()
    img = np.array(img)

    return img


def centered_rotation_matrix(alpha, center):
    R = np.array([
        [math.cos(alpha), -math.sin(alpha), 0],
        [math.sin(alpha), math.cos(alpha), 0],
        [0, 0, 1]
    ])
    C1 = np.eye(3)
    C1[0, 2] = -center[0]
    C1[1, 2] = -center[1]
    C2 = np.eye(3)
    C2[0, 2] = center[0]
    C2[1, 2] = center[1]

    return np.dot(C2, np.dot(R, C1))


def randomLetter(letters_set=gs.LETTERS, rotated=False):

    font = random.choice(gs.FONTS)
    letter = random.choice(letters_set)

    img = drawLetter(letter, size=100, font=font)

    M = np.eye(3)
    M[2, 2] = 100/gs.PATCH_SIZE[0]
    M[..., :2] += np.random.uniform(low=-0.001, high=0.001, size=(3, 2))

    alpha = random.uniform(-math.pi/18, math.pi/18)
    if rotated:
        alpha += random.uniform(math.pi/4, math.pi*7/4)
    R = centered_rotation_matrix(alpha, np.array(gs.PATCH_SIZE)/2)
    img = cv2.warpPerspective(img, np.dot(R, M), dsize=gs.PATCH_SIZE)

    return img, letters_set.index(letter)
