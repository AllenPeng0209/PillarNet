import torch



def compute_locations(features, fpn_strides):
    locations = []
    for level, feature in enumerate(features):
        h, w = feature.size()[-2:]
        locations_per_level = compute_locations_per_level(
           h, w, fpn_strides[level],
           feature.device
           )
        locations.append(locations_per_level)
    return locations

def compute_locations_per_level(h, w, stride, device):
    shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
    shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations
