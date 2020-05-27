import numpy as np
import os
import cv2
import math
import time
from tqdm import tqdm
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from src.config import config

folder_path = "bin/output/"

# Transformation between electrode coords and pixels
H_green = np.array([
    [-0.000859855242, 1.516164163, -773.447535],
    [-1.514683557, -0.0006185068021, 744.8603579],
    [-0.000002604686097, 0.000001519154879, 1.0]
])
H_red = np.array([
    [-0.0008421953675, 1.515467887, -760.717246],
    [-1.508434614, -0.001225029098, 756.2824433],
    [-0.0000004530558222, -0.00000391697885, 1.0]
])

pause_replay = False
slow_replay = False


def mouse_callback(event, x, y, flags, param):
    global pause_replay, slow_replay
    if event == cv2.EVENT_LBUTTONDOWN:
        pause_replay = not pause_replay
        print(f"Replay paused: {pause_replay}")
    if event == cv2.EVENT_RBUTTONDOWN:
        slow_replay = not slow_replay
        print(f"Replay slowed: {slow_replay}")


def save_image_figure(img, file_name, legend=None, legend_size=8):
    plt.imshow(img)
    fig = plt.gca()
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if legend:
        legend_handles = []
        for i, leg in enumerate(legend):
            legend_handles.append(mpatches.Patch(color=config.colors[i] / 255.0, label=leg))
        plt.legend(handles=legend_handles, loc=1, prop={'size': legend_size})
    plt.savefig(file_name, format='eps', dpi=300, bbox_inches='tight',
                transparent="True", pad_inches=0.0)


def draw_circle_path(img, H, radius, color, thickness, end_angle=2 * math.pi):
    angle = 0
    inv_H = np.linalg.inv(H)
    while angle < end_angle:
        new_angle = angle + math.pi / 24.0  # 5 deg resolution
        start_p = (np.dot(inv_H, (np.array([radius * math.cos(angle), radius * math.sin(angle), 1.0]))))
        start_p = (start_p / start_p[2]).astype(np.int)
        end_p = (np.dot(inv_H, (np.array([radius * math.cos(new_angle), radius * math.sin(new_angle), 1.0]))))
        end_p = (end_p / end_p[2]).astype(np.int)
        img = cv2.line(img, (start_p[1], start_p[0]), (end_p[1], end_p[0]), color, thickness)
        angle = new_angle


def replay_data_original(file_name, start_frame=0, end_frame=0):
    cap = cv2.VideoCapture(os.path.join(folder_path, file_name + '.mp4'))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    data = np.loadtxt(os.path.join(folder_path, file_name + '.csv'), delimiter=',').astype(np.int)
    data[:, 0] -= data[0, 0]  # Start time from 0

    index = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    cv2.namedWindow('replay', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('replay', 512, 512)
    cv2.setMouseCallback('replay', mouse_callback)

    success, img = cap.read()
    while index < end_frame:
        while pause_replay:
            cv2.waitKey(100)

        # CUSTOM PART
        bbox = data[index, 1:5]
        # draw_circle_path(img, H_red, 550, colors[1], 4)
        img = cv2.rectangle(img,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            config.colors[2],
                            2)
        # img = cv2.rectangle(img,
        #                     (pos_x - bbox_size - 1, pos_y - bbox_size - 1),
        #                     (pos_x + bbox_size - 1, pos_y + bbox_size - 1),
        #                     (255, 255, 0),
        #                     2)

        path_history = data[start_frame:index + 1, 7:9]
        for i in range(1, path_history.shape[0]):
            p_start = (path_history[i - 1, 0], path_history[i - 1, 1])
            p_end = (path_history[i, 0], path_history[i, 1])
            cv2.line(img, p_start, p_end, config.colors[0], 3)

        img = cv2.putText(img, f"Time: {data[index, 0]:.0f} ms |  Index: {index - start_frame} | Data ID: {index}",
                          (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (255, 0, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('replay', img)
        cv2.waitKey(1)

        if slow_replay:
            time.sleep(0.1)

        success, img = cap.read()
        index += 1

    cv2.waitKey(0)


def replay_data_repaired_time(file_name, start_frame=0, end_frame=-1, fps=30, write_output=False):
    fps_step = 1.0 / fps
    cap = cv2.VideoCapture(os.path.join(folder_path, file_name + '.mp4'))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if write_output:
        out = cv2.VideoWriter(os.path.join(folder_path, file_name + '_fixed.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps,
                              (1024, 1024))
    data = np.loadtxt(os.path.join(folder_path, file_name + '.csv'), delimiter=',').astype(np.int)
    data[:, 0] -= data[0, 0]  # Start time from 0

    cv2.namedWindow('replay', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('replay', 512, 512)
    cv2.setMouseCallback('replay', mouse_callback)

    sim_frame_id = 0
    sim_time = data[start_frame, 0]
    sim_start_t = data[start_frame, 0]
    sim_end_t = data[end_frame, 0]
    data_index = 0
    for i in tqdm(range(int((sim_end_t - sim_start_t) / (1000 * fps_step)))):
        start_time = time.time()
        data_index = np.argmin(np.abs(data[:, 0] - sim_time))
        cap.set(cv2.CAP_PROP_POS_FRAMES, data_index)
        success, img = cap.read()
        if not success:
            print(f"Error reading video, frame={data_index}")
            break
        bbox = data[data_index, 1:5]
        # pos_x, pos_y = data[data_index, 5:7]
        # pos_x_min, pos_y_min = data[data_index, 7:9]

        # radius = 550
        # period = 50
        # t = (i*fps_step)/period
        # angle = math.pi / 8.
        # inv_H = np.linalg.inv(H_green)
        # while angle > (-t*2*math.pi + math.pi/8.):
        #     new_angle = angle - math.pi / 1800.0  # 5 deg resolution
        #     start_p = (np.dot(inv_H, (np.array([radius * math.cos(angle), radius * math.sin(angle), 1.0]))))
        #     start_p = (start_p / start_p[2]).astype(np.int)
        #     end_p = (np.dot(inv_H, (np.array([radius * math.cos(new_angle), radius * math.sin(new_angle), 1.0]))))
        #     end_p = (end_p / end_p[2]).astype(np.int)
        #     img = cv2.line(img, (start_p[1], start_p[0]), (end_p[1], end_p[0]), config.colors[1], 4)
        #     angle = new_angle
        # draw_circle_path(img, H_red, 550, (0, 0, 255), 4)
        img = cv2.rectangle(img,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            config.colors[2],
                            2)
        # img = cv2.rectangle(img,
        #                     (pos_x - bbox_size - 1, pos_y - bbox_size - 1),
        #                     (pos_x + bbox_size - 1, pos_y + bbox_size - 1),
        #                     (255, 255, 0),
        #                     2)

        path_history = data[start_frame:data_index + 1, 7:9]
        for i in range(1, path_history.shape[0]):
            p_start = (path_history[i - 1, 0], path_history[i - 1, 1])
            p_end = (path_history[i, 0], path_history[i, 1])
            cv2.line(img, p_start, p_end, config.colors[0], 3)
        # img = cv2.putText(img, f"Time: {sim_time:.0f} ms |  Id: {sim_frame_id} | Data: {data_index}", (20, 20),
        #                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (255, 0, 0), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow('replay', img)
        if write_output:
            out.write(img)
        cv2.waitKey(1)

        sim_time += (fps_step * 1000)
        sim_frame_id += 1
        # print(f"Time computing one frame: {1000 * (time.time() - start_time)}")

    if write_output:
        out.release()
    cv2.waitKey(10)


def plot_path_results(file_name, segments):
    cap = cv2.VideoCapture(os.path.join(folder_path, file_name + '.mp4'))
    data = np.loadtxt(os.path.join(folder_path, file_name + '.csv'), delimiter=',').astype(np.int)
    data[:, 0] -= data[0, 0]  # Start time from 0

    background_frame = segments[0][1]
    if background_frame < 0:
        background_frame = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - abs(background_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, background_frame)  # read end frame of first segment as background
    success, img = cap.read()

    # CUSTOM PART
    # draw_circle_path(img, H_red, 550, colors[1], 5)

    legend = []
    for index, seg in enumerate(segments):
        start_frame, end_frame = seg[0:2]
        path_history = data[start_frame:end_frame, 5:7]
        for i in range(1, path_history.shape[0]):
            p_start = (path_history[i - 1, 0], path_history[i - 1, 1])
            p_end = (path_history[i, 0], path_history[i, 1])
            cv2.line(img, p_start, p_end, config.colors[seg[2]], 5)
        legend.append(seg[3])
        print(f"Time for segment {index}: {data[end_frame, 0] - data[start_frame, 0]} ms")

    bbox = data[segments[0][1], 1:5]
    img = cv2.rectangle(img,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        config.colors[2],
                        4)

    save_image_figure(img, os.path.join(folder_path, file_name + '_result.eps'), legend)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(os.path.join(folder_path, file_name + '_multi.png'), img)
    # cv2.waitKey(1)


if __name__ == "__main__":
    # file_name = '2020_05_05_16_21_52_optim_vs_direct'
    # plot_path_results(file_name, [(1370, 1558, 0, "Optimal"), (2095, 2332, 1, "Direct")])

    file_name = '2020_05_14_15_52_40_o0_c0_550um_radius'
    plot_path_results(file_name, [(200, -1, 0, "Measured"), (0, 0, 1, "Reference")])

    # file_name = '2020_05_18_16_12_32_o0_c0'
    # plot_path_results(file_name, [(1390, 1550, 0, "P-reg, no trim"), (2110, 2380, 1, "P-reg, trim")])

    # [(start_frame, end_frame, color_id, label), ...]
    # compare_multiple_paths(file_name, [(1370, 1558, "Optimal"), (2095, 2332, "Direct")])

    # Experiment with circular trajectory
    # file_name = '2020_05_14_15_52_40_o0_c0_550um_radius'
    # replay_data_repaired_time(file_name, start_frame=220, end_frame=-10, fps=15, write_output=True)
    # file_name = '2020_05_14_15_52_40_o0_c1_550um_radius'
    # replay_data_repaired_time(file_name, start_frame=190, end_frame=-53, fps=15, write_output=True)

    # file_name = '2020_05_05_16_21_52_optim_vs_direct'
    ## replay_data_repaired_time(file_name, start_frame=1370, end_frame=-872, fps=30, write_output=False)
    # replay_data_repaired_time(file_name, start_frame=2095, end_frame=-99, fps=30, write_output=True)

    # replay_data_original(file_name, offset_frame=1100, skip_last_frames=0)
