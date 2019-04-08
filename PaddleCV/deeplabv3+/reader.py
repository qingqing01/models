from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image, ImageFilter
import numpy as np
import random
import cv2
import os
import six
import math

default_config = {
    "shuffle": True,
    "min_resize": 0.5,
    "max_resize": 4,
    "crop_size": 769,
    "min_scale_factor": 0.5,
    "max_scale_factor": 2.,
    "scale_factor_step_size": 0.25,
    "mean_pixel": [127.5, 127.5, 127.5],
    "ignore_label": 255,
    "norm": False,
}

img_mean = np.array([0.485, 0.456, 0.406]).reshape(
    (1, 3, 1, 1)).astype(np.float32)
img_std = np.array([0.229, 0.224, 0.225]).reshape(
    (1, 3, 1, 1)).astype(np.float32)


def slice_with_pad(a, s, value=0):
    pads = []
    slices = []
    for i in range(len(a.shape)):
        if i >= len(s):
            pads.append([0, 0])
            slices.append([0, a.shape[i]])
        else:
            l, r = s[i]
            if l < 0:
                pl = -l
                l = 0
            else:
                pl = 0
            if r > a.shape[i]:
                pr = r - a.shape[i]
                r = a.shape[i]
            else:
                pr = 0
            pads.append([pl, pr])
            slices.append([l, r])
    slices = list(map(lambda x: slice(x[0], x[1], 1), slices))
    a = a[slices]
    a = np.pad(a, pad_width=pads, mode='constant', constant_values=value)
    return a


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """Gets a random scale value.
  Args:
    min_scale_factor (float): Minimum scale value.
    max_scale_factor (float): Maximum scale value.
    step_size: The step size from minimum to maximum value.
  Returns:
    A random scale value selected between minimum and maximum value.
  Raises:
    ValueError: min_scale_factor has unexpected value.
  """
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return min_scale_factor

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return np.random.uniform(min_scale_factor, max_scale_factor, size=1)

    # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = np.linspace(min_scale_factor, max_scale_factor, num_steps)
    np.random.shuffle(scale_factors)
    return scale_factors[0]


def resize_to_range(image,
                    label=None,
                    min_size=None,
                    max_size=None,
                    factor=None):
    """
    1. If the image can be rescaled so its minimum size is equal to min_size
       without the other side exceeding max_size, then do so.
    2. Otherwise, resize so the largest side is equal to max_size.

    Args:
      image: A 3D tensor of shape [height, width, channels].
      label: (optional) A 3D tensor of shape [height, width, channels].
      min_size: (scalar) desired size of the smaller image side.
      max_size: (scalar) maximum allowed size of the larger image side. Note
        that the output dimension is no larger than max_size and may be slightly
        smaller than min_size when factor is not None.
      factor: Make output size multiple of factor plus one.

    """
    new_tensor_list = []
    if max_size is not None:
        if factor is not None:
            max_size = max_size + (factor -
                                   (max_size - 1) % factor) % factor - factor
    orig_height, orig_width, channel = image.shape
    orig_min_size = min(orig_height, orig_width)
    large_scale_factor = float(min_size) / float(orig_min_size)

    large_height = int(math.ceil(orig_height * large_scale_factor))
    large_width = int(math.ceil(orig_width * large_scale_factor))

    # cv2.resize(im, (width, height))
    large_size = np.stack([large_width, large_height])

    new_size = large_size
    if max_size is not None:
        # Calculate the smaller of the possible sizes, use that if the larger
        # is too big.
        orig_max_size = max(orig_height, orig_width)
        small_scale_factor = float(max_size) / float(orig_max_size)
        small_height = int(math.ceil(orig_height * small_scale_factor))
        small_width = int(math.ceil(orig_width * small_scale_factor))

        small_size = np.stack([small_width, small_height])
        new_size = small_size if max(large_size) > max_size else large_size

    # Ensure that both output sides are multiples of factor plus one.
    if factor is not None:
        new_size += (factor - (new_size - 1) % factor) % factor

    new_size = new_size.tolist()
    new_tensor_list.append(
        cv2.resize(
            image, new_size, interpolation=cv2.INTER_CUBIC))

    if label is not None:
        # Input label has shape [height, width, channel].
        resized_label = cv2.resize(label, new_size)
        new_tensor_list.append(resized_label)
    else:
        new_tensor_list.append(None)
    return new_tensor_list


def randomly_scale_image_and_label(image, label=None, scale=1.0):
    """Randomly scales image and label.
    Args:
      image: Image with shape [height, width, 3].
      label: Label with shape [height, width, 1].
      scale: The value to scale image and label.
    Returns:
      Scaled image and label.
    """
    # No random scaling if scale == 1.
    if scale == 1.0:
        return image, label

    image_shape = image.shape  # height, width, channel
    new_dim = np.stack(
        [image_shape[1], image_shape[0]]).astype(np.float32) * scale
    new_dim = new_dim.astype(np.int32)

    new_dim = new_dim.tolist()
    new_shape = (new_dim[0], new_dim[1])
    image = cv2.resize(image, new_shape, interpolation=cv2.INTER_CUBIC)
    if label is not None:
        label = cv2.resize(label, new_shape)
        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=2)

    return image, label


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_value):
    """Pads the given image with the given pad_value.
    Works like tf.image.pad_to_bounding_box, except it can pad the image
    with any given arbitrary pad value and also handle images whose sizes are not
    known during graph construction.
    Args:
      image: 3-D tensor with shape [height, width, channels]
      offset_height: Number of rows of zeros to add on top.
      offset_width: Number of columns of zeros to add on the left.
      target_height: Height of output image.
      target_width: Width of output image.
      pad_value: Value to pad the image tensor with.
    Returns:
      3-D tensor of shape [target_height, target_width, channels].
    Raises:
      ValueError: If the shape of image is incompatible with the offset_* or
      target_* arguments.
    """
    image -= pad_value
    height, width = image.shape[0:2]
    assert target_height >= height
    assert target_width >= width
    after_padding_height = target_height - offset_height - height
    after_padding_width = target_width - offset_width - width
    assert after_padding_height >= 0
    assert after_padding_width >= 0

    height_params = np.stack([offset_height, after_padding_height])
    width_params = np.stack([offset_width, after_padding_width])
    channel_params = np.stack([0, 0])
    paddings = np.stack([height_params, width_params, channel_params])
    padded = np.pad(image, paddings, mode='constant', constant_values=0)
    outputs = padded + pad_value
    return outputs


def random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.
    The function applies the same crop to each image in the list. This can be
    effectively applied when there are multiple image inputs of the same
    dimension such as:
      image, depths, normals = random_crop([image, depths, normals], 120, 150)
    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the new height.
      crop_width: the new width.
    Returns:
      the image_list with cropped images.
    Raises:
      ValueError: if there are multiple image inputs provided with different size
        or the images are smaller than the crop dimensions.
    """
    if not image_list:
        raise ValueError('Empty image_list.')

    image_shape = image_list[0].shape
    image_height = image_shape[0]
    image_width = image_shape[1]
    assert image_height >= crop_height
    assert image_width >= crop_width

    for i in range(1, len(image_list)):
        image = image_list[i]
        shape = image.shape
        height = shape[0]
        width = shape[1]
        assert height == image_height
        assert width == image_width

    max_offset_height = image_height - crop_height + 1
    max_offset_width = image_width - crop_width + 1

    # [0, max_offset_height)
    offset_height = np.random.randint(max_offset_height)
    offset_width = np.random.randint(max_offset_width)

    return [
        image[offset_height:offset_height + crop_height, offset_width:
              offset_width + crop_width, :] for image in image_list
    ]


class CityscapeDataset:
    def __init__(self,
                 dataset_dir,
                 subset='train',
                 config=default_config,
                 tf_reader=False):
        label_dirname = os.path.join(dataset_dir, 'gtFine/' + subset)
        if six.PY2:
            import commands
            label_files = commands.getoutput(
                "find %s -type f | grep labelTrainIds | sort" %
                label_dirname).splitlines()
        else:
            import subprocess
            label_files = subprocess.getstatusoutput(
                "find %s -type f | grep labelTrainIds | sort" %
                label_dirname)[-1].splitlines()
        self.label_files = label_files
        self.label_dirname = label_dirname
        self.index = 0
        self.subset = subset
        self.dataset_dir = dataset_dir
        self.config = config
        self.tf_reader = tf_reader
        self.reset()
        print("total number", len(label_files))

    def reset(self, shuffle=False):
        self.index = 0
        if self.config["shuffle"]:
            np.random.shuffle(self.label_files)

    def next_img(self):
        self.index += 1
        if self.index >= len(self.label_files):
            self.reset()

    def color_augmentation(self, img):
        r = np.random.rand()
        img = img / 255.0
        if r < 0.5:
            img = np.power(img, 1 - r)
        else:
            img = np.power(img, 1 + (r - 0.5) * 2)
        img = img * 255.0
        return img

    def get_img_tf(self):
        shape = self.config["crop_size"]
        while True:
            ln = self.label_files[self.index]
            img_name = os.path.join(
                self.dataset_dir,
                'leftImg8bit/' + self.subset + ln[len(self.label_dirname):])
            img_name = img_name.replace('gtFine_labelTrainIds', 'leftImg8bit')
            label = np.expand_dims(cv2.imread(ln, cv2.IMREAD_GRAYSCALE), axis=2)
            img = cv2.imread(img_name).astype(np.float32)
            if img is None:
                print("load img failed:", img_name)
                self.next_img()
            else:
                break
        if shape == -1:
            return img, label, ln

        min_resize_value = self.config['min_resize']
        max_resize_value = self.config['max_resize']
        min_scale_factor = self.config['min_scale_factor']
        max_scale_factor = self.config['max_scale_factor']
        scale_factor_step_size = self.config['scale_factor_step_size']

        processed_image = img
        if min_resize_value or max_resize_value:
            [processed_image, label] = resize_to_range(
                image=processed_image,
                label=label,
                min_size=min_resize_value,
                max_size=max_resize_value)

        scale = get_random_scale(min_scale_factor, max_scale_factor,
                                 scale_factor_step_size)
        processed_image, label = randomly_scale_image_and_label(processed_image,
                                                                label, scale)

        # Pad image and label to have dimensions >= [crop_height, crop_width]
        image_shape = processed_image.shape
        image_height, image_width = image_shape[0:2]

        crop_height, crop_width = self.config['crop_size'], self.config[
            'crop_size']
        target_height = image_height + max(crop_height - image_height, 0)
        target_width = image_width + max(crop_width - image_width, 0)

        # Pad image with mean pixel value.
        mean_pixel = self.config['mean_pixel']
        mean_pixel = np.reshape(np.array(mean_pixel), [1, 1, 3])
        processed_image = pad_to_bounding_box(
            processed_image, 0, 0, target_height, target_width, mean_pixel)

        if label is not None:
            ignore_label = self.config['ignore_label']
            label = pad_to_bounding_box(label, 0, 0, target_height,
                                        target_width, ignore_label)

        # Randomly crop the image and label.
        if label is not None:
            processed_image, label = random_crop([processed_image, label],
                                                 crop_height, crop_width)

        # Randomly left-right flip the image and label.
        if np.random.rand() > 0.5:
            processed_image = processed_image[:, ::-1, :]
            label = label[:, ::-1, :]
        return processed_image, label, []

    def get_img(self):
        shape = self.config["crop_size"]
        while True:
            ln = self.label_files[self.index]
            img_name = os.path.join(
                self.dataset_dir,
                'leftImg8bit/' + self.subset + ln[len(self.label_dirname):])
            img_name = img_name.replace('gtFine_labelTrainIds', 'leftImg8bit')
            label = cv2.imread(ln)
            img = cv2.imread(img_name)
            if img is None:
                print("load img failed:", img_name)
                self.next_img()
            else:
                break
        if shape == -1:
            return img, label, ln

        if np.random.rand() > 0.5:
            range_l = 1
            range_r = self.config['max_resize']
        else:
            range_l = self.config['min_resize']
            range_r = 1

        if np.random.rand() > 0.5:
            assert len(img.shape) == 3 and len(
                label.shape) == 3, "{} {}".format(img.shape, label.shape)
            img = img[:, ::-1, :]
            label = label[:, ::-1, :]
        #img = self.color_augmentation(img)

        random_scale = np.random.rand(1) * (range_r - range_l) + range_l
        crop_size = int(shape / random_scale)
        bb = crop_size // 2

        def _randint(low, high):
            return int(np.random.rand(1) * (high - low) + low)

        offset_x = np.random.randint(bb, max(bb + 1, img.shape[0] -
                                             bb)) - crop_size // 2
        offset_y = np.random.randint(bb, max(bb + 1, img.shape[1] -
                                             bb)) - crop_size // 2
        img_crop = slice_with_pad(img, [[offset_x, offset_x + crop_size],
                                        [offset_y, offset_y + crop_size]], 128)
        img = cv2.resize(img_crop, (shape, shape))
        label_crop = slice_with_pad(label, [[offset_x, offset_x + crop_size],
                                            [offset_y, offset_y + crop_size]],
                                    255)
        label = cv2.resize(
            label_crop, (shape, shape), interpolation=cv2.INTER_NEAREST)
        return img, label, ln + str(
            (offset_x, offset_y, crop_size, random_scale))

    def get_img_enc(self):
        shape = self.config["crop_size"]
        while True:
            ln = self.label_files[self.index]
            img_name = os.path.join(
                self.dataset_dir,
                'leftImg8bit/' + self.subset + ln[len(self.label_dirname):])
            img_name = img_name.replace('gtFine_labelTrainIds', 'leftImg8bit')
            label = cv2.imread(ln)
            img = cv2.imread(img_name)
            if img is None:
                print("load img failed:", img_name)
                self.next_img()
            else:
                break
        if shape == -1:
            return img, label, ln

        if np.random.rand() > 0.5:
            range_l = 1
            range_r = self.config['max_resize']
        else:
            range_l = self.config['min_resize']
            range_r = 1

        random_scale = np.random.rand(1) * (range_r - range_l) + range_l
        crop_size = int(shape / random_scale)
        bb = crop_size // 2

        def _randint(low, high):
            return int(np.random.rand(1) * (high - low) + low)

        offset_x = np.random.randint(bb, max(bb + 1, img.shape[0] -
                                             bb)) - crop_size // 2
        offset_y = np.random.randint(bb, max(bb + 1, img.shape[1] -
                                             bb)) - crop_size // 2
        img_crop = slice_with_pad(img, [[offset_x, offset_x + crop_size],
                                        [offset_y, offset_y + crop_size]],
                                  127.5)
        img = cv2.resize(
            img_crop, (shape, shape), interpolation=cv2.INTER_CUBIC)
        label_crop = slice_with_pad(label, [[offset_x, offset_x + crop_size],
                                            [offset_y, offset_y + crop_size]],
                                    255)
        label = cv2.resize(
            label_crop, (shape, shape), interpolation=cv2.INTER_NEAREST)

        if np.random.rand() > 0.5:
            img = img[:, :, (2, 1, 0)]
            img = Image.fromarray(img)
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            img = np.array(img).astype('float32')[:, :, (2, 1, 0)]

        if np.random.rand() > 0.5:
            assert len(img.shape) == 3 and len(
                label.shape) == 3, "{} {}".format(img.shape, label.shape)
            img = img[:, ::-1, :]
            label = label[:, ::-1, :]

        return img, label, ln + str(
            (offset_x, offset_y, crop_size, random_scale))

    def get_batch(self, batch_size=1):
        imgs = []
        labels = []
        names = []
        while len(imgs) < batch_size:
            img, label, ln = self.get_img_tf(
            ) if self.tf_reader else self.get_img_enc()
            #) if self.tf_reader else self.get_img()
            imgs.append(img)
            labels.append(label)
            names.append(ln)
            self.next_img()
        return np.array(imgs), np.array(labels), names

    def get_batch_generator(self, batch_size, total_step):
        def do_get_batch():
            for i in range(total_step):
                imgs, labels, names = self.get_batch(batch_size)
                labels = labels.astype(np.int32)[:, :, :, 0]
                if self.config['norm']:
                    imgs = imgs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
                    imgs -= img_mean
                    imgs /= img_std
                else:
                    imgs = imgs.transpose(0, 3, 1, 2).astype(np.float32) / (
                        255.0 / 2) - 1
                    #imgs = imgs[:,:,:,::-1].transpose(0, 3, 1, 2).astype(np.float32) / (
                    #    255.0 / 2) - 1
                yield i, imgs, labels, names

        batches = do_get_batch()
        try:
            from prefetch_generator import BackgroundGenerator
            batches = BackgroundGenerator(batches, 100)
        except:
            print(
                "You can install 'prefetch_generator' for acceleration of data reading."
            )
        return batches


if __name__ == '__main__':
    dataset_path = "/home/users/dangqingqing/data/cityscapes/"
    default_config['min_resize'] = None
    default_config['max_resize'] = None
    dataset = CityscapeDataset(dataset_path, 'train', tf_reader=True)
    batches = dataset.get_batch_generator(4, 20)
    _, imgs, labels, names = next(batches)
