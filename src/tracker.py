import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import time
from torch.autograd import Variable

from .alexnet import SiameseAlexNet
from .config import config
from .utils import get_exemplar_image, get_pyramid_instance_image, get_instance_image, crop_and_pad


class ToTensor(object):
    def __call__(self, sample):
        sample = sample.transpose(2, 0, 1)
        return torch.from_numpy(sample.astype(np.float32))


class SiamFCTrackerNoScale:
    """
    Simplified version of the SiamFC tracker that does NOT use scaling.
    It means that the bounding box does not change in size, but it is only translated.
    """

    def __init__(self, model_path, gpu_id):
        self.gpu_id = gpu_id
        with torch.cuda.device(gpu_id):
            self.model = SiameseAlexNet(gpu_id)
            self.model.load_state_dict(torch.load(model_path))
            self.model = self.model.cuda()
            self.model.eval()
        self.transforms = transforms.Compose([
            ToTensor()
        ])

    def _cosine_window(self, size):
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):
        """ Initialize the tracker.
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        self.bbox = (bbox[0] - 1, bbox[1] - 1, bbox[0] - 1 + bbox[2], bbox[1] - 1 + bbox[3])  # zero based
        self.pos = np.array(
            [bbox[0] - 1 + (bbox[2] - 1) / 2, bbox[1] - 1 + (bbox[3] - 1) / 2])  # center x, center y, zero based
        self.target_sz = np.array([bbox[2], bbox[3]])  # width, height
        # get exemplar img
        self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        self.exemplar_img, scale_z, s_z = get_exemplar_image(frame, self.bbox,
                                                             config.exemplar_size, config.context_amount, self.img_mean)

        # get exemplar feature
        exemplar_img = self.transforms(self.exemplar_img)[None, :, :, :]
        with torch.cuda.device(self.gpu_id):
            exemplar_img_var = Variable(exemplar_img.cuda())
            self.model((exemplar_img_var, None))

        # create cosine window
        self.interp_response_sz = config.response_up_stride * config.response_sz
        self.cosine_window = self._cosine_window((self.interp_response_sz, self.interp_response_sz))

        # create s_x
        self.s_x = s_z + (config.instance_size - config.exemplar_size) / scale_z

    def update(self, frame, time_benchmark=None, debug=False):
        """track object based on the previous frame
        Args:
            frame: an RGB image

        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        start_time = time.time()
        instance_img = crop_and_pad(frame, self.pos[0], self.pos[1], config.instance_size, self.s_x, self.img_mean)
        if debug:
            instance_img_copy = np.copy(instance_img)
        if time_benchmark is not None:
            time_benchmark["crop_and_pad"].append(time.time() - start_time)
        if config.visualisation and config.num_of_objects == 1 and config.num_of_channels == 1:
            cv2.imshow('instance_img', instance_img)
            cv2.imshow('exemplar', self.exemplar_img)
            cv2.waitKey(1)

        instance_img = self.transforms(instance_img)[None, :, :, :]
        with torch.cuda.device(self.gpu_id):
            start_time = time.time()
            instance_img_var = Variable(instance_img.cuda())

            if time_benchmark is not None:
                torch.cuda.synchronize(self.gpu_id)
                time_benchmark["transfer_to_gpu"].append(time.time() - start_time)
                start_time = time.time()

            response_map = self.model((None, instance_img_var))

            if time_benchmark is not None:
                torch.cuda.synchronize(self.gpu_id)
                time_benchmark["cnn"].append(time.time() - start_time)
                start_time = time.time()

            response_map = response_map.data.cpu().numpy().squeeze()

            if debug:
                response_map_original = np.copy(response_map)
            if time_benchmark is not None:
                torch.cuda.synchronize(self.gpu_id)
                time_benchmark["transfer_to_cpu"].append(time.time() - start_time)
                start_time = time.time()

            response_map = cv2.resize(response_map, (self.interp_response_sz, self.interp_response_sz), cv2.INTER_CUBIC)

            if time_benchmark is not None:
                time_benchmark["interpolation"].append(time.time() - start_time)

        start_time = time.time()
        # penalty scale change
        response_map -= response_map.min()
        response_map /= response_map.sum()
        response_map = (1 - config.window_influence) * response_map + \
                       config.window_influence * self.cosine_window

        if config.visualisation and config.num_of_objects == 1 and config.num_of_channels == 1:
            cv2.imshow('response_map', response_map / np.max(response_map))
            cv2.waitKey(1)

        max_r, max_c = np.unravel_index(response_map.argmax(), response_map.shape)
        # displacement in interpolation response
        disp_response_interp = np.array([max_c, max_r]) - (self.interp_response_sz - 1) / 2.
        # displacement in input
        disp_response_input = disp_response_interp * config.total_stride / config.response_up_stride
        # displacement in frame
        disp_response_frame = disp_response_input * self.s_x / config.instance_size
        # position in frame coordinates
        self.pos += disp_response_frame
        bbox = (self.pos[0] - self.target_sz[0] / 2 + 1,  # xmin   convert to 1-based
                self.pos[1] - self.target_sz[1] / 2 + 1,  # ymin
                self.pos[0] + self.target_sz[0] / 2 + 1,  # xmax
                self.pos[1] + self.target_sz[1] / 2 + 1)  # ymax
        if time_benchmark is not None:
            time_benchmark["displacement"].append(time.time() - start_time)

        if debug:
            return bbox, instance_img_copy, self.exemplar_img, response_map_original, response_map, max_r, max_c
        else:
            return bbox


class SiamFCTracker:
    def __init__(self, model_path, gpu_id):
        self.gpu_id = gpu_id
        with torch.cuda.device(gpu_id):
            self.model = SiameseAlexNet(gpu_id)
            self.model.load_state_dict(torch.load(model_path))
            self.model = self.model.cuda()
            self.model.eval() 
        self.transforms = transforms.Compose([
            ToTensor()
        ])

    def _cosine_window(self, size):
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):
        """ Initialize the tracker.
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        self.bbox = (bbox[0]-1, bbox[1]-1, bbox[0]-1+bbox[2], bbox[1]-1+bbox[3]) # zero based
        self.pos = np.array([bbox[0]-1+(bbox[2]-1)/2, bbox[1]-1+(bbox[3]-1)/2])  # center x, center y, zero based
        self.target_sz = np.array([bbox[2], bbox[3]])                            # width, height
        # get exemplar img
        self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        exemplar_img, scale_z, s_z = get_exemplar_image(frame, self.bbox,
                config.exemplar_size, config.context_amount, self.img_mean)

        # get exemplar feature
        exemplar_img = self.transforms(exemplar_img)[None,:,:,:]
        with torch.cuda.device(self.gpu_id):
            exemplar_img_var = Variable(exemplar_img.cuda())
            self.model((exemplar_img_var, None))

        self.penalty = np.ones((config.num_scale)) * config.scale_penalty
        self.penalty[config.num_scale//2] = 1

        # create cosine window
        self.interp_response_sz = config.response_up_stride * config.response_sz
        self.cosine_window = self._cosine_window((self.interp_response_sz, self.interp_response_sz))

        # create scalse
        self.scales = config.scale_step ** np.arange(np.ceil(config.num_scale/2)-config.num_scale,
                np.floor(config.num_scale/2)+1)

        # create s_x
        self.s_x = s_z + (config.instance_size-config.exemplar_size) / scale_z

        # arbitrary scale saturation
        self.min_s_x = 0.2 * self.s_x
        self.max_s_x = 5 * self.s_x

    def update(self, frame):
        """ Track the object based on the previous frame.
        Args:
            frame: an RGB image

        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        size_x_scales = self.s_x * self.scales
        pyramid = get_pyramid_instance_image(frame, self.pos, config.instance_size, size_x_scales, self.img_mean)
        if config.visualisation:
            for i, im in enumerate(pyramid):
                cv2.imshow(f'instance_img_{i}', im)
                cv2.waitKey(1)
        instance_imgs = torch.cat([self.transforms(x)[None,:,:,:] for x in pyramid], dim=0)
        with torch.cuda.device(self.gpu_id):
            instance_imgs_var = Variable(instance_imgs.cuda())
            response_maps = self.model((None, instance_imgs_var))
            response_maps = response_maps.data.cpu().numpy().squeeze()
            response_maps_up = [cv2.resize(x, (self.interp_response_sz, self.interp_response_sz), cv2.INTER_CUBIC)
             for x in response_maps]
        # get max score
        max_score = np.array([x.max() for x in response_maps_up]) * self.penalty

        # penalty scale change
        scale_idx = max_score.argmax()
        response_map = response_maps_up[scale_idx]
        response_map -= response_map.min()
        response_map /= response_map.sum()
        response_map = (1 - config.window_influence) * response_map + \
                config.window_influence * self.cosine_window
        if config.visualisation:
            cv2.imshow(f'response_map', response_map)
            cv2.waitKey(1)
        max_r, max_c = np.unravel_index(response_map.argmax(), response_map.shape)
        # displacement in interpolation response
        disp_response_interp = np.array([max_c, max_r]) - (self.interp_response_sz-1) / 2.
        # displacement in input
        disp_response_input = disp_response_interp * config.total_stride / config.response_up_stride
        # displacement in frame
        scale = self.scales[scale_idx]
        disp_response_frame = disp_response_input * (self.s_x * scale) / config.instance_size
        # position in frame coordinates
        self.pos += disp_response_frame
        # scale damping and saturation
        self.s_x *= ((1 - config.scale_lr) + config.scale_lr * scale)
        self.s_x = max(self.min_s_x, min(self.max_s_x, self.s_x))
        self.target_sz = ((1 - config.scale_lr) + config.scale_lr * scale) * self.target_sz
        bbox = (self.pos[0] - self.target_sz[0]/2 + 1, # xmin   convert to 1-based
                self.pos[1] - self.target_sz[1]/2 + 1, # ymin
                self.pos[0] + self.target_sz[0]/2 + 1, # xmax
                self.pos[1] + self.target_sz[1]/2 + 1) # ymax
        return bbox
