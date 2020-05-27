import os
import numpy as np
import cv2
import matplotlib
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from src.tracker import SiamFCTrackerNoScale, config
from src.utils import find_object_center

data_folder = 'bin/output/'

gpu_id = 0
selected_position = (0, 0)


def save_img_fig(img, filename, title, xlabel, ylabel, cmap=None, legend=False):
    font = {'size': 12}
    matplotlib.rc('font', **font)
    plt.imshow(img, cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        blue = mpatches.Patch(color=config.colors[0] / 255.0, label='Orig BBox')
        red = mpatches.Patch(color=config.colors[1] / 255.0, label='Improved method')
        plt.legend(handles=[blue, red])
    plt.savefig(filename, bbox_inches='tight', transparent="True", pad_inches=0.01)
    plt.show()


def save_center_test(img_first, img_second, bbox_first, bbox_second, bbox_size):
    pos_x_first, pos_y_first = find_object_center(img_first, bbox_first, 'region_min')
    cv2.rectangle(img_first, ((bbox_first[0]), (bbox_first[1])), ((bbox_first[2]), (bbox_first[3])), config.colors[0],
                  3)
    cv2.rectangle(img_first, (int(pos_x_first - bbox_size / 2), int(pos_y_first - bbox_size / 2)),
                  (int(pos_x_first + bbox_size / 2), int(pos_y_first + bbox_size / 2)), config.colors[1], 2)
    save_img_fig(img_first[pos_y_first - 127:pos_y_first + 127, pos_x_first - 127:pos_x_first + 127],
                 os.path.join(data_folder, 'demo_bbox_first.eps'), 'First frame',
                 'x [px]', 'y [px]', None, True)
    pos_x, pos_y = find_object_center(img_second, bbox_second, 'region_min')
    cv2.rectangle(img_second, ((bbox_second[0]), (bbox_second[1])), ((bbox_second[2]), (bbox_second[3])),
                  config.colors[0], 3)
    cv2.rectangle(img_second, (int(pos_x - bbox_size / 2), int(pos_y - bbox_size / 2)),
                  (int(pos_x + bbox_size / 2), int(pos_y + bbox_size / 2)), config.colors[1], 2)
    save_img_fig(img_second[pos_y_first - 127:pos_y_first + 127, pos_x_first - 127:pos_x_first + 127],
                 os.path.join(data_folder, 'demo_bbox_second.eps'), 'Second frame',
                 'x [px]', 'y [px]', None, True)


def generate_images(video_path, index_first, index_second):
    """
    Assumes second index is bigger
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index_first)
    success, img = cap.read()
    cv2.imwrite(os.path.join(data_folder, 'demo_single_first.png'), img)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index_second)
    success, img = cap.read()
    cv2.imwrite(os.path.join(data_folder, 'demo_single_second.png'), img)


# mouse callback function
def mouse_callback(event, x, y, flags, param):
    global selected_position
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_position = (x, y)
        print(x, y)


if __name__ == "__main__":
    """
    Frames indices have to be selected before
    """
    generate_images(os.path.join(data_folder, '2020_05_04_17_45_07_o0_c0.mp4'), 360, 370)

    img_first = cv2.imread(os.path.join(data_folder, 'demo_single_first.png'))
    img_second = cv2.imread(os.path.join(data_folder, 'demo_single_second.png'))
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', img_first.shape[1], img_first.shape[0])
    cv2.setMouseCallback('image', mouse_callback)
    cv2.imshow('image', img_first)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    BBOX_SIZE = 50
    bbox = (selected_position[0] - BBOX_SIZE / 2, selected_position[1] - BBOX_SIZE / 2, BBOX_SIZE, BBOX_SIZE)
    tracker = SiamFCTrackerNoScale(config.model_path, gpu_id)
    tracker.model.eval()

    tracker.init(img_first, bbox)
    bbox = (bbox[0] - 1, bbox[1] - 1,
            bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1)

    save_img_fig(img_first, os.path.join(data_folder, 'demo_single_first.eps'), 'First frame',
                 'x [px]', 'y [px]')
    bbox_first, instance, exemplar, response_map_original, response_map, max_r, max_c = tracker.update(img_first,
                                                                                                       debug=True)
    bbox_first = np.round(bbox_first).astype(np.int)
    save_img_fig(instance, os.path.join(data_folder, 'demo_single_instance_first.eps'), 'First instance frame',
                 'x [px]', 'y [px]')
    save_img_fig(exemplar, os.path.join(data_folder, 'demo_single_exemplar.eps'), 'Exemplar', 'x [px]', 'y [px]')

    bbox_second, instance, exemplar, response_map_original, response_map, max_r, max_c = tracker.update(img_second,
                                                                                                        debug=True)
    bbox_second = np.round(bbox_second).astype(np.int)
    response_map_original -= response_map_original.min()
    response_map_original /= response_map_original.sum()
    save_img_fig(img_second, os.path.join(data_folder, 'demo_single_second.eps'), 'First frame',
                 'x [px]', 'y [px]')
    save_img_fig(instance, os.path.join(data_folder, 'demo_single_instance_second.eps'), 'Second instance frame',
                 'x [px]', 'y [px]')
    save_img_fig(cv2.resize(response_map_original, (272, 272), fx=0, fy=0, interpolation=cv2.INTER_NEAREST),
                 os.path.join(data_folder, 'demo_single_response_original.eps'), 'Score map - NN interpolation',
                 'x [px]', 'y [px]', cmap='gray')
    save_img_fig(response_map, os.path.join(data_folder, 'demo_single_response.eps'),
                 'Score map - Bicubic interpolation', 'x [px]', 'y [px]', cmap='gray')
    save_center_test(img_first, img_second, bbox_first, bbox_second, BBOX_SIZE)
    cv2.destroyAllWindows()
