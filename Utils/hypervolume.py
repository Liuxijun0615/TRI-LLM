import numpy as np


def calculate_hypervolume(points, reference_point):

    points = np.array(points)
    reference_point = np.array(reference_point)


    if len(points) == 0:
        return 0.0


    if points.ndim == 1:
        points = points.reshape(1, -1)


    valid_mask = np.all(points >= reference_point, axis=1)
    points = points[valid_mask]

    if len(points) == 0:
        return 0.0


    n_obj = points.shape[1]

    if n_obj == 2:
        return _hv_2d(points, reference_point)
    else:

        return _hv_nd(points, reference_point)


def _hv_2d(points, ref):

    points = points[np.argsort(-points[:, 0])]

    volume = 0.0
    current_max_y = ref[1]

    for pt in points:

        if pt[1] > current_max_y:

            width = pt[0] - ref[0]

            height = pt[1] - current_max_y

            volume += width * height

            current_max_y = pt[1]

    return volume


def _hv_nd(points, ref):

    n_points, n_dim = points.shape

    if n_dim == 2:
        return _hv_2d(points, ref)


    z_values = np.unique(points[:, -1])

    z_values = np.sort(z_values)[::-1]


    z_values = np.concatenate([z_values, [ref[-1]]])

    z_values = z_values[z_values >= ref[-1]]

    z_values = np.unique(z_values)[::-1]

    volume = 0.0


    for i in range(len(z_values) - 1):
        z_high = z_values[i]
        z_low = z_values[i + 1]


        height = z_high - z_low
        if height <= 0:
            continue


        active_points_mask = points[:, -1] >= z_high
        active_points = points[active_points_mask, :-1]
        if len(active_points) > 0:

            area_nd_minus_1 = calculate_hypervolume(active_points, ref[:-1])


            volume += area_nd_minus_1 * height

    return volume



def hypervolume(objectives, ref_point=None):
    if ref_point is None:

        ref_point = np.zeros(objectives.shape[1])
    return calculate_hypervolume(objectives, ref_point)


def normalized_hypervolume(points, ideal=None, nadir=None):

    if points.ndim == 1:
        points = points.reshape(1, -1)
    return calculate_hypervolume(points, np.zeros(points.shape[1]))