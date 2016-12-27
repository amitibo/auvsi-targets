from __future__ import division
import numpy as np
import scipy.cluster as spc
import scipy.ndimage.morphology as spm
import cv2

import AUVSItargets.global_settings as gs

VISUALIZE = False


def innerMost(R, shape_colors_map, shape_indices):
    mean_0 = R.ravel()[shape_indices[shape_colors_map == 0]].mean()
    mean_1 = R.ravel()[shape_indices[shape_colors_map == 1]].mean()
    if mean_0 < mean_1:
        inner_index = 0
    else:
        inner_index = 1

    return inner_index


def getLetterMask_scipy(img):
    """Calculate the mask """

    #
    # Calculate distance map.
    #
    w, h = img.shape[:2]
    X, Y = np.mgrid[:h, :w]
    R = np.sqrt((X-w/2)**2 + (Y-h/2)**2)

    #
    # Divide the crop to two colors.
    #
    img_vectors = img.reshape((-1, 3)).astype(np.float)
    colors, dist = spc.vq.kmeans(img_vectors, 2)
    color_map, _ = spc.vq.vq(img_vectors, colors)

    #
    # Assuming a tight crop the target should occupy most of the crop.
    #
    img_indices = np.arange(w*h)
    shape_index = innerMost(R, color_map, img_indices)

    #
    # Smooth the map to remove noise.
    # Erode it to remove the border of the shape that might damage the
    # segmentation to shape and letter.
    #
    shape_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    shape_mask.ravel()[color_map == shape_index] = 1
    shape_mask = cv2.medianBlur(shape_mask, ksize=15)
    kernel = np.ones((5, 5), np.uint8)
    shape_mask = cv2.erode(shape_mask, kernel, iterations=1)

    #
    # Fill holes (incase the color of the letter is similar to the background.)
    #
    shape_mask = spm.binary_fill_holes(shape_mask)

    #
    #
    #
    shape_indices = np.arange(shape_mask.size)[shape_mask.ravel() == 1]

    #
    # Split the shape in two colors.
    #
    shape_vectors = img_vectors[shape_mask.ravel() == 1]
    shape_colors, dist = spc.vq.kmeans(shape_vectors, 2)
    shape_colors_map, _ = spc.vq.vq(shape_vectors, shape_colors)
    logging.info('Classified shape colors: {}'.format(shape_colors))

    #
    # The letter should be the innermost
    #
    letter_index = innerMost(R, shape_colors_map, shape_indices)
    logging.info('Classified letter color index: {}'.format(letter_index))

    #
    # Letter indices
    #
    letter_indices = shape_indices[shape_colors_map == letter_index]
    neto_shape_indices = shape_indices[shape_colors_map == (1-letter_index)]
    letter_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    letter_mask.ravel()[letter_indices] = 255
    neto_shape_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    neto_shape_mask.ravel()[neto_shape_indices] = 0  # ATODO

    letter_mask = cv2.erode(letter_mask, kernel, iterations=1)

    return letter_mask, colors


def calcKMeans(points, K):
    """Calculate KMeans."""

    if len(points) < K:
        return False, None, None

    term_crit = (cv2.TERM_CRITERIA_EPS, 30, 0.1)
    if cv2.__version__[0] == '3':
        #
        # The interface is slightly different in the case of cv2 version 3.
        #
        ret, labels, centers = cv2.kmeans(points, K, None, term_crit, 10, 0)
    else:
        ret, labels, centers = cv2.kmeans(points, K, term_crit, 10, 0)

    #
    # Evaluate success.
    # Note:
    # If one of the segments is less than 5% we assume
    # that the K was too big and therefore failed.
    #
    for i in range(K):
        if (labels==i).sum()/labels.size < 0.05:
            return False, None, None

    return True, labels, centers


def getLetterMask_cv2(crop):
    #
    # Calculate K kmeans: bg color, fg color, letter color
    # I use the LAB color space.
    #
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab_points = lab.reshape(-1, 3).astype(np.float32)

    K = 4
    while K > 1:
        success, labels, centers = calcKMeans(lab_points, K)
        if success:
            break
        K -= 1

    if not success:
        return (None, None)

    #
    # Use the normalised moments to identify the fg (shape) mask.
    # The target is in the center and has the minimum normalized
    # moments of order nu20, nu02.
    #
    kernel = np.ones((3, 3), np.uint8)
    mom = []
    bin_imgs = []
    for i in range(K):
        bin_img = np.zeros(shape=crop.shape[:2], dtype=np.uint8)
        bin_img.flat = labels == i
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        bin_imgs.append(bin_img)
        moments = cv2.moments(bin_img, True)
        mom.append((moments['nu20'] + moments['nu02'])*moments['m00'])

    order = np.argsort(mom)

    if VISUALIZE:
        for i, o in enumerate(order):
            win_name = 'bin {}'.format(i)
            cv2.namedWindow(win_name, flags=cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, bin_imgs[o]*255)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #
    # The shape is uaually the second in the order
    # but just in case, we verify that its area is
    # bigger as somtimes the letter color matches
    # a very noisy surrounding.
    #
    if cv2.__version__[0] == '3':
        #
        # It seems that in version 3.1.0 (and possibly other >3 versions) the
        # findContours function returns: binary_img, contours, hierarchy tuple.
        #
        _, contours, hierarchy = cv2.findContours(
            bin_imgs[order[0]].copy(),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        contours, hierarchy = cv2.findContours(
            bin_imgs[order[0]].copy(),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
    if len(contours) == 0:
        area0 = 0
    else:
        contours0 = max(contours, key=cv2.contourArea)
        area0 = cv2.contourArea(contours0)

    if cv2.__version__[0] == '3':
        #
        # It seems that in version 3.1.0 (and possibly other >3 versions) the
        # findContours function returns: binary_img, contours, hierarchy tuple.
        #
        _, contours, hierarchy = cv2.findContours(
            bin_imgs[order[1]].copy(),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        contours, hierarchy = cv2.findContours(
            bin_imgs[order[1]].copy(),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
    if len(contours) == 0:
        area1 = 0
    else:
        contours1 = max(contours, key=cv2.contourArea)
        area1 = cv2.contourArea(contours1)

    if area0 > area1:
        shape_index = 0
    else:
        shape_index = 1

    shape = bin_imgs[order[shape_index]]

    #
    # Fill the letter hole in the shape.
    # Note:
    # I first try to connect broken
    # lines
    #
    kernel = np.ones((3, 3), np.uint8)
    shape = cv2.filter2D(shape, -1, kernel)
    shape[shape>0] = 1
    if cv2.__version__[0] == '3':
        _, contours, hierarchy = cv2.findContours(
            shape.copy(),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        contours, hierarchy = cv2.findContours(
            shape.copy(),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
    max_contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(shape)
    cv2.fillPoly(filled, [np.squeeze(max_contour)], (1,))
    filled = cv2.erode(filled, kernel, iterations=3)

    #
    # Segment the shape to 2 colors: fg and letter
    #
    fg_indices = filled.flat == 1
    success, labels, centers = calcKMeans(lab_points[fg_indices], K=2)
    if not success:
        return (None, None)

    #
    # Use the number of pixels to identify the fg mask.
    # Usually the letter is smaller than the shape.
    #
    bin_imgs = []
    moments = []
    for i in range(2):
        bin_img = np.zeros(shape=crop.shape[:2], dtype=np.uint8)
        bin_img.flat[fg_indices] = labels == i
        bin_imgs.append(bin_img)

        m = cv2.moments(bin_img, True)
        moments.append(m['m00'])

    #
    # Identify the letter, fg masks
    #
    colors = np.squeeze(
        cv2.cvtColor(centers.reshape(2, 1, 3).astype(np.uint8),
            cv2.COLOR_LAB2RGB)
    )

    letter_index = np.argmin(moments)
    letter_mask = bin_imgs[letter_index]

    if letter_index == 1:
        colors = colors[::-1, ...]

    letter_mask = letter_mask*255

    return letter_mask, colors


def getLetterMask(img):
    if gs.USE_CV2_KMEANS:
        return getLetterMask_cv2(img)
    else:
        return getLetterMask_scipy(img)
