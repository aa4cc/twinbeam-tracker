import numpy as np
import cv2
from scipy.ndimage.filters import minimum_filter


def get_center(x):
    return (x - 1.) / 2.


def xyxy2cxcywh(bbox):
    return get_center(bbox[0] + bbox[2]), \
           get_center(bbox[1] + bbox[3]), \
           (bbox[2] - bbox[0]), \
           (bbox[3] - bbox[1])


def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    xmin = cx - original_sz // 2
    xmax = cx + original_sz // 2
    ymin = cy - original_sz // 2
    ymax = cy + original_sz // 2
    im_h, im_w, _ = img.shape

    left = right = top = bottom = 0
    if xmin < 0:
        left = int(abs(xmin))
    if xmax > im_w:
        right = int(xmax - im_w)
    if ymin < 0:
        top = int(abs(ymin))
    if ymax > im_h:
        bottom = int(ymax - im_h)

    xmin = int(max(0, xmin))
    xmax = int(min(im_w, xmax))
    ymin = int(max(0, ymin))
    ymax = int(min(im_h, ymax))
    im_patch = img[ymin:ymax, xmin:xmax]
    if left != 0 or right != 0 or top != 0 or bottom != 0:
        if img_mean is None:
            img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=img_mean)
    if model_sz != original_sz:
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    return im_patch


def get_exemplar_image(img, bbox, size_z, context_amount, img_mean=None):
    cx, cy, w, h = xyxy2cxcywh(bbox)
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z
    exemplar_img = crop_and_pad(img, cx, cy, size_z, s_z, img_mean)
    return exemplar_img, scale_z, s_z


def get_instance_image(img, bbox, size_z, size_x, context_amount, img_mean=None):
    cx, cy, w, h = xyxy2cxcywh(bbox)
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z
    d_search = (size_x - size_z) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    scale_x = size_x / s_x
    instance_img = crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
    return instance_img, scale_x, s_x


def get_pyramid_instance_image(img, center, size_x, size_x_scales, img_mean=None):
    if img_mean is None:
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
    pyramid = [crop_and_pad(img, center[0], center[1], size_x, size_x_scale, img_mean)
               for size_x_scale in size_x_scales]
    return pyramid


def find_object_center(img, bbox, method='bbox_center', type='spherical'):
    """

    :param img:
    :param bbox:
    :param type:
    :param method:
    :return:
    """
    if method == 'region_min':
        try:
            context_add = 25
            blur = cv2.GaussianBlur(img[(bbox[1] - context_add):(bbox[3] + context_add),
                                    (bbox[0] - context_add):(bbox[2]) + context_add, 0].astype(np.float),
                                    (29, 29), 127)
            selection = blur[context_add:-context_add, context_add:-context_add]
            lm = minimum_filter(blur, footprint=np.ones((3, 3)))
            msk = (blur == lm)
            msk = msk[context_add:-context_add, context_add:-context_add]
            min_indices = np.argwhere(msk)
            min_index = min_indices[np.argmin(selection[min_indices[:, 0], min_indices[:, 1]])]
            return (np.round(bbox[0]) + min_index[1] - 1), (np.round(bbox[1]) + min_index[0] - 1)
        except Exception:
            print("find_object_center(regional_min) failed, fallback to bbox center")
            return find_object_center(img, bbox, 'bbox_center')
    else:
        return int(np.round((bbox[0] + bbox[2]) / 2)), int(np.round((bbox[1] + bbox[3]) / 2))
