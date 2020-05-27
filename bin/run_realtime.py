import os
import numpy as np
import cv2
import time
from datetime import datetime
from collections import deque
import sys

sys.path.append(os.getcwd())

import socket
from _thread import *
import threading
from threading import Thread
from struct import pack
import time
import sys

from src.tracker import SiamFCTrackerNoScale
from src.config import config
from src.utils import find_object_center

print('Using CV2: ' + cv2.__version__)

# Used for indexing
CHANNEL_GREEN = 0
CHANNEL_RED = 1
POS_X = 0
POS_Y = 1

image_lock = threading.Lock()
ctl_lock = threading.Lock()
initialized_images = False
initialized_tracker = np.zeros((config.num_of_objects, config.num_of_channels), dtype=np.bool)
initialized_target = np.zeros((config.num_of_objects, config.num_of_channels), dtype=np.bool)

initial_bbox = np.zeros((config.num_of_objects, config.num_of_channels, 4,), dtype=np.int)
positions = np.zeros((config.num_of_objects, config.num_of_channels, 2), dtype=np.int)
tracker_iterations = np.uint32(0)
images = np.zeros((config.num_of_channels, config.image_size_y, config.image_size_x, 3), dtype=np.uint8)

gpu_id = 0


def request_image(soc, command, size):
    data = b''
    counter = 0
    soc.send(command)
    while 1:
        part = soc.recv(16384)
        part_size = len(part)
        if not part:
            print("[img-thread] data null")
            break
        # print(f"[img-thread] received new img, size {len(data)}")
        counter += part_size
        data += part
        if counter >= size:
            if counter > size:
                print(f'[img-thread] Warning: received size bigger ({counter}) than expected {size}')
            return data
    return None


def keyboard_thread():
    try:
        for line in sys.stdin:
            if 'v' in line:
                config.visualisation = not config.visualisation
                print(f"[keyboard] Toggled visualisation: {config.visualisation}")
            elif 'd' in line:
                config.debug = not config.debug
                print(f"[keyboard] Toggled debug: {config.debug}")
    except KeyboardInterrupt:
        print("[keyboard] Ended by KeyboardInterrupt")


def image_thread():
    global initialized_images, images, image_lock
    print("[img-thread] Starting socket")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    while 1:
        try:
            s.connect((config.image_socket_ip, config.image_socket_port))
            break
        except ConnectionRefusedError:
            print("[img-thread] Connection refused, camera image service not running. Retrying...")
            time.sleep(1)

    # print('[img-thread] Connection address:', conn)

    DESIRED_LOOP_TIME = 1 / 30
    time_data_size = 10
    time_data = deque(maxlen=time_data_size)
    time_log_index = 0
    start_time = time.time()
    try:
        while 1:
            # command
            # 0 green backprop
            # 1 red backprop
            # 2 green raw
            # 3 red raw
            # 4 both images backprop
            # 5 both images raw
            command = pack('cB', b'r', config.channel)

            size_single = config.image_size_y * config.image_size_x
            size = size_single * config.num_of_channels

            img_data = request_image(s, command, size)
            if img_data is None:
                print("[img-thread] Received None, ending image thread.")
                break

            image_lock.acquire()
            for i in range(config.num_of_channels):
                images[i] = np.stack((np.frombuffer(img_data, np.uint8, size_single, i * size_single).reshape(
                    (config.image_size_y, config.image_size_x)),) * 3, axis=-1)
            image_lock.release()
            if config.debug:
                if time_log_index == 120:  # approx every 2 sec
                    time_log_index = 0
                    average_time = np.average(time_data)
                    print(f'[img-thread] Avg time loading (last {time_data_size} val): {average_time * 1000:.2f}ms '
                          f'| FPS: {(1 / average_time):.1f}')
                else:
                    time_log_index += 1
            if not initialized_images:
                print("[img-thread] Received first image, init done.")
                initialized_images = True

            loop_duration = time.time() - start_time
            if loop_duration < DESIRED_LOOP_TIME:
                time.sleep(DESIRED_LOOP_TIME - loop_duration)
            loop_duration = time.time() - start_time
            start_time = time.time()
            time_data.append(loop_duration)
    except KeyboardInterrupt:
        print("[img-thread] KeyboardInterrupt")
    s.close()
    print("[img-thread] Ended")


def tracker_init_thread():
    global initialized_images, positions, ctl_lock
    print("[tracker-init-thread] Starting...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((config.control_socket_ip, config.tracker_init_socket_port))
    s.listen()
    print("[tracker-init-thread] Listening...")
    try:
        while 1:
            conn, addr = s.accept()
            print(f'[tracker-init-thread] Connected clint: {addr}')
            while 1:
                part = conn.recv(16384)
                part_size = len(part)
                if not part:
                    print(f"[tracker-init-thread] {addr} disconnected")
                    break

                ctl_lock.acquire()
                try:
                    prefix = chr(part[0]) + chr(part[1])
                    if prefix == 'ti':
                        offset = 6
                        for channel_id in range(config.num_of_channels):
                            for obj_id in range(config.num_of_objects):
                                pos_x = np.frombuffer(part, np.uint16, 1, offset)
                                offset += 2
                                pos_y = np.frombuffer(part, np.uint16, 1, offset)
                                offset += 2
                                half_size = config.bb_size / 2
                                initial_bbox[obj_id, channel_id] = np.array([
                                    pos_x - half_size, pos_y - half_size, config.bb_size, config.bb_size
                                ], dtype=np.int)
                                positions[obj_id, channel_id, POS_X] = pos_x
                                positions[obj_id, channel_id, POS_X] = pos_y
                                initialized_tracker[obj_id, channel_id] = False
                                initialized_target[obj_id, channel_id] = True
                                if config.debug:
                                    print(
                                        f"[tracker-init-thread] Initialized tracker {obj_id}/{channel_id} with position: [{pos_x}, {pos_y}]")
                except Exception:
                    pass
                ctl_lock.release()
    except KeyboardInterrupt:
        print("[tracker-init-thread] KeyboardInterrupt")
    s.close()
    print("[tracker-init-thread] Ended")


def control_thread():
    # 16 bit num of green, 16 bit num of red,
    # then xy both 16 unsigned short for green, then xy both 16 unsigned short for red
    global initialized_images, positions, ctl_lock, tracker_iterations
    print("[ctl-thread] Starting...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((config.control_socket_ip, config.control_socket_port))
    s.listen()
    print("[ctl-thread] Listening...")

    try:
        while 1:
            conn, addr = s.accept()
            print(f'[ctl-thread] Connected clint: {addr}')
            while 1:
                part = conn.recv(16384)
                part_size = len(part)
                if not part:
                    print(f"[ctl-thread] {addr} disconnected")
                    break

                ctl_lock.acquire()
                pos = np.copy(positions)
                targets_initialized = np.all(initialized_target)
                ctl_lock.release()

                if targets_initialized:
                    data = b''
                    data += pack('I', np.uint32(tracker_iterations))
                    data += pack('H', np.uint16(config.num_of_objects))  # Num of green
                    data += pack('H', np.uint16(config.num_of_objects))  # Num of red
                    for channel_id in range(config.num_of_channels):
                        for obj_id in range(config.num_of_objects):
                            data += pack('HH', np.uint16(pos[obj_id, channel_id, POS_X]),
                                         np.uint16(pos[obj_id, channel_id, POS_Y]))
                    conn.sendall(data)
    except KeyboardInterrupt:
        print("[ctl-thread] KeyboardInterrupt")
    s.close()
    print("[ctl-thread] Ended")


def tracker_thread(obj_id, channel_id):
    global image_lock, images, ctl_lock, tracker_iterations
    t = threading.current_thread()
    t.alive = True

    datestr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if config.write_data_output_to_file:
        data_output_file = os.path.join('bin/output/', f"{datestr}_o{obj_id}_c{channel_id}.csv")
        datafile = open(data_output_file, 'w')
        datafile.write(
            "# timestamp (ms),bbox[0],bbox[1],bbox[2],bbox[3],bbox_center_x,bbox_center_y,region_min_x,region_min_y\n")
    if config.write_video_output_to_file:
        video_output_file = os.path.join('bin/output/', f"{datestr}_o{obj_id}_c{channel_id}.mp4")
        out = cv2.VideoWriter(video_output_file, cv2.VideoWriter_fourcc(*'avc1'), 30,
                              (config.image_size_y, config.image_size_x))
    # starting tracking
    tracker = SiamFCTrackerNoScale(config.model_path, gpu_id)

    while not initialized_target[obj_id, channel_id] and t.alive:
        time.sleep(0.1)

    print(f"[Tracker {obj_id}/{channel_id}] Target initialized, starting tracking")
    time_data = np.zeros((100), dtype=np.float)
    time_log_index = 0
    while t.alive:
        start_time = time.time()
        image_lock.acquire()
        img = np.copy(images[channel_id])
        image_lock.release()
        if not initialized_tracker[obj_id, channel_id]:
            bbox = initial_bbox[obj_id, channel_id]
            tracker.init(img, bbox)
            bbox = np.round((bbox[0] - 1, bbox[1] - 1,
                             bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1)).astype(np.int)
            initialized_tracker[obj_id, channel_id] = True
        else:
            bbox = np.round(tracker.update(img)).astype(np.int)

        pos_x1, pos_y1 = find_object_center(img, bbox, method='region_min')
        pos_x, pos_y = find_object_center(None, bbox, method='bbox_center')
        ctl_lock.acquire()
        positions[obj_id, channel_id, POS_X] = pos_x1
        positions[obj_id, channel_id, POS_Y] = pos_y1
        if obj_id == 0 and channel_id == 0:
            tracker_iterations += 1
        ctl_lock.release()

        if config.debug:
            time_data[time_log_index] = (time.time() - start_time) * 1000
            if time_log_index == 99:
                time_log_index = 0
                avg_time = np.average(time_data)
                print(
                    f"[Tracker {obj_id}/{channel_id}] Average time (100 values) per frame : {avg_time:.2f}ms | FPS: {(1.0 / (avg_time / 1000)):.1f}")
            else:
                time_log_index += 1

        write_time = time.time()
        if config.write_data_output_to_file:
            datafile.write(
                f"{int(write_time * 1000)},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{pos_x},{pos_y},{pos_x1},{pos_y1}\n")
            # if config.debug:
            #    print(f"[Tracker {obj_id}/{channel_id}] Time writing to disk: {(time.time() - write_time) * 1000:.2f} ms")
        if config.write_video_output_to_file:
            out.write(img)
            cv2.waitKey(1)

        if config.visualisation:
            # bbox xmin ymin xmax ymax
            img = cv2.rectangle(img,
                                (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])),
                                (0, 255, 0) if channel_id == CHANNEL_GREEN else (255, 0, 0),
                                2)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow(f'Tracking {obj_id}/{channel_id}', cv2.resize(img, (512, 512)))
            cv2.waitKey(1)
    if config.write_data_output_to_file:
        datafile.close()
    if config.write_video_output_to_file:
        out.release()
        cv2.waitKey(1)
    print(f"[Tracker {obj_id}/{channel_id}] Released resources")
    cv2.destroyAllWindows()
    print(f"[Tracker {obj_id}/{channel_id}] ended")


def select_tracked_object(image, obj_id, channel_id):
    bbox = cv2.selectROI(f"Select object:  id={obj_id}, channel={channel_id}", image, showCrosshair=True)
    cv2.destroyAllWindows()
    cv2.waitKey(250)
    return bbox


def main():
    global initial_bbox, image_lock, initialized_tracker, initialized_images

    print("##### Starting tracker #####")
    print('Press "d" to toggle DEBUG logs')
    print('Press "v" to toggle VISUALIZATION')

    start_new_thread(keyboard_thread, ())

    print("Starting image thread...")
    start_new_thread(image_thread, ())

    try:
        while not initialized_images:
            time.sleep(0.5)
        if not config.use_remote_target_setting:
            for obj_id in range(config.num_of_objects):
                for channel_id in range(config.num_of_channels):
                    np.copyto(initial_bbox[obj_id, channel_id],
                              select_tracked_object(images[channel_id], obj_id, channel_id))
                    initialized_target[obj_id, channel_id] = True
    except KeyboardInterrupt:
        exit()

    start_new_thread(tracker_init_thread, ())
    start_new_thread(control_thread, ())

    threads = []
    for obj_id in range(config.num_of_objects):
        for channel_id in range(config.num_of_channels):
            t = Thread(target=tracker_thread, args=(obj_id, channel_id))
            t.start()
            threads.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for t in threads:
            t.alive = False
            t.join()
        exit()


if __name__ == "__main__":
    main()
