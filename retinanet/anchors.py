import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = torch.tensor([0.5, 1, 2])
        else:
            self.ratios = ratios
        if scales is None:
            self.scales = torch.tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        else:
            self.scales = scales

    def forward(self, image):
        
        image_shape = image.shape[2:]
        image_shape = torch.tensor(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        scales = self.scales.to(image.device)
        ratios = self.ratios.to(image.device)

        # compute anchors over all pyramid levels
        all_anchors = torch.zeros((0, 4), device=image.device)

        for idx, p in enumerate(self.pyramid_levels):
            anchors         = generate_anchors(base_size=self.sizes[idx], ratios=ratios, scales=scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors     = torch.cat((all_anchors, shifted_anchors), dim=0)

        all_anchors = torch.unsqueeze(all_anchors, 0)

        return all_anchors

def generate_anchors(base_size: int, ratios, scales):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = torch.tensor([0.5, 1, 2])

    if scales is None:
        scales = torch.tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = torch.zeros((num_anchors, 4), device=ratios.device)

    # scale base_size
    anchors[:, 2:] = base_size * torch.transpose(scales.repeat(2, len(ratios)), 0, 1)

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = (areas / ratios.repeat_interleave(len(scales))).sqrt()
    anchors[:, 3] = anchors[:, 2] * ratios.repeat_interleave(len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= torch.transpose((anchors[:, 2] * 0.5).repeat(2, 1), 0, 1)
    anchors[:, 1::2] -= torch.transpose((anchors[:, 3] * 0.5).repeat(2, 1), 0, 1)

    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride: int, anchors):
    shift_x = (torch.arange(0, shape[1], device=anchors.device) + 0.5) * stride
    shift_y = (torch.arange(0, shape[0], device=anchors.device) + 0.5) * stride

    shift_y, shift_x = torch.meshgrid(shift_y, shift_x)

    shifts = torch.stack([
        torch.reshape(shift_x, (-1,)), torch.reshape(shift_y, (-1,)),
        torch.reshape(shift_x, (-1,)), torch.reshape(shift_y, (-1,))
    ]).transpose(0, 1)

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).permute((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

