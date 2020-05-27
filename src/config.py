import numpy as np


class Config:
    # Tracking realtime settings
    visualisation = False
    debug = False
    write_data_output_to_file = True
    write_video_output_to_file = False
    use_remote_target_setting = True
    image_socket_ip = '147.32.86.138'
    image_socket_port = 30000
    control_socket_ip = '147.32.86.138'
    control_socket_port = 30002
    tracker_init_socket_port = 30003
    image_size_x = 1024
    image_size_y = 1024
    bb_size = 50  # in pixels, for automatic target BBox setup
    # 0 green backprop
    # 1 red backprop
    # 2 green raw
    # 3 red raw
    # 4 both images backprop
    # 5 both images raw
    channel = 4
    num_of_objects = 1
    num_of_channels = 1 if channel <= 3 else 2
    model_path = './models/baseline-conv5_e55.pth'

    # dataset related
    exemplar_size = 127  # exemplar size
    instance_size = 255  # instance size
    context_amount = 0.5  # context amount

    # training related
    num_per_epoch = 53200  # num of samples per epoch
    train_ratio = 0.9  # training ratio of VID dataset
    frame_range = 100  # frame range of choosing the instance
    train_batch_size = 8  # training batch size
    valid_batch_size = 8  # validation batch size
    train_num_workers = 8  # number of workers of train dataloader
    valid_num_workers = 8  # number of workers of validation dataloader
    lr = 1e-2  # learning rate of SGD
    momentum = 0.0                         # momentum of SGD
    weight_decay = 0.0                     # weight decay of optimizator
    step_size = 25                         # step size of LR_Schedular
    gamma = 0.1                            # decay rate of LR_Schedular
    epoch = 30                             # total epoch
    seed = 1234                            # seed to sample training videos
    log_dir = './models/logs'              # log dirs
    radius = 16                            # radius of positive label
    response_scale = 1e-3                  # normalize of response
    max_translate = 3                      # max translation of random shift

    # tracking related
    scale_step = 1.0375                    # scale step of instance image
    num_scale = 3                          # number of scales
    scale_lr = 0.59  # scale learning rate
    response_up_stride = 16  # response upsample stride
    response_sz = 17  # response size
    train_response_sz = 15  # train response size
    window_influence = 0.176  # window influence
    scale_penalty = 0.9745  # scale penalty
    total_stride = 8  # total stride of backbone
    sample_type = 'uniform'
    gray_ratio = 0.25
    blur_ratio = 0.15

    # Visualization
    colors = (np.array([
        [9, 132, 227],
        [214, 48, 49],
        [254, 211, 48]
    ])).astype(np.float)

config = Config()
