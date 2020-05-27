import os
import numpy as np
import cv2
import time
from datetime import datetime
from fire import Fire
from tqdm import tqdm
from src.tracker import SiamFCTrackerNoScale
from src.utils import find_object_center
from src.config import config

gpu_id = 0
output_folder = 'bin/output/'


def select_tracked_object(image):
    bbox = cv2.selectROI("Select object for tracking", image, showCrosshair=True)
    cv2.destroyAllWindows()
    cv2.waitKey(250)
    return bbox


def main(video_path, model_path, write_output, visualization=False, benchmark=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    datestr = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    video_output_file = datestr + '.mp4'
    data_output_file = datestr + '.csv'
    print(f"--- Running video: {video_path} ---")
    print(f"FPS: {fps}")
    print(f"Frames: {frame_count}")
    print(f"Write output: {write_output}")
    print(f"Output file: {datestr}[.mp4/.csv]")

    if benchmark:
        time_benchmark = {
            "crop_and_pad": [],
            "transfer_to_gpu": [],
            "cnn": [],
            "transfer_to_cpu": [],
            "interpolation": [],
            "displacement": []
        }
    else:
        time_benchmark = None

    success, img = cap.read()

    init = False
    # starting tracking
    tracker = SiamFCTrackerNoScale(model_path, gpu_id)
    tracker.model.eval()
    if write_output:
        datafile = open(os.path.join(output_folder, data_output_file), 'w')
        datafile.write(
            "# timestamp (ms),bbox[0],bbox[1],bbox[2],bbox[3],bbox_center_x,bbox_center_y,region_min_x,region_min_y\n")
        out = cv2.VideoWriter(os.path.join(output_folder, video_output_file), cv2.VideoWriter_fourcc(*'avc1'), fps,
                              (img.shape[1], img.shape[0]))

    try:
        for i in tqdm(range(frame_count)):
            if not init:
                bbox = select_tracked_object(img)
                tracker.init(img, bbox)
                bbox = (bbox[0] - 1, bbox[1] - 1,
                        bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1)
                init = True
            else:
                bbox = np.round(tracker.update(img, time_benchmark)).astype(np.int)
                # bbox: [xmin, ymin, xmax, ymax]

            pos_x1, pos_y1 = find_object_center(img, bbox, method='region_min')
            pos_x2, pos_y2 = find_object_center(img, bbox, method='bbox_center')
            img = cv2.rectangle(img,
                                ((bbox[0]), (bbox[1])),
                                ((bbox[2]), (bbox[3])),
                                config.colors[1],
                                3)

            img = cv2.putText(img, f"{i}/{frame_count}", (30, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                              config.colors[2], 2)
            img = cv2.putText(img, f"Pos: [{pos_x2},{pos_y2}]", (30, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                              config.colors[1], 2)
            # img = cv2.circle(img, (pos_x1, pos_y1), 1, (0, 0, 255), 1)
            # img = cv2.circle(img, (pos_x1, pos_y1), 30, (0, 0, 255), 1)
            # img = cv2.circle(img, (pos_x2, pos_y2), 1, (255, 0, 0), 1)
            # img = cv2.circle(img, (pos_x2, pos_y2), 30, (255, 0, 0), 1)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if visualization:
                cv2.imshow(f'{video_output_file}', cv2.resize(img, (1024, 1024)))
                cv2.waitKey(1)
            if write_output:
                datafile.write(
                    f"{int(time.time() * 1000)},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{pos_x2},{pos_y2},{pos_x1},{pos_y1}\n")
                out.write(img)
                cv2.waitKey(1)
            success, img = cap.read()
    except KeyboardInterrupt:
        pass

    cv2.waitKey(1)

    if write_output:
        datafile.close()
        out.release()

    if benchmark:
        datafile = open(os.path.join(output_folder, datestr + "_benchmark.csv"), 'w')
        datafile.write("# ")
        for labels in time_benchmark:
            datafile.write(f"{labels},")
        datafile.write("\n")
        for i in range(len(time_benchmark["crop_and_pad"]) - 1):
            line = ""
            for labels in time_benchmark:
                line += f"{int(time_benchmark[labels][i] * 1e6)},"
            line = line[:-1]
            line += "\n"
            datafile.write(line)
        datafile.close()


if __name__ == "__main__":
    Fire(main)
