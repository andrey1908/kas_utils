import sys
import numpy as np
from scipy.ndimage import minimum_filter
from .utils import get_depth_scale

try:
    import torch
except ImportError:
    pass


class DepthToPointCloud:
    def __init__(self, K, D, pool_size=8, return_named_point_cloud=False,
            backend="scipy"):
        assert backend in ["scipy", "torch"]

        self.K = K
        self.D = D
        self.pool_size = pool_size
        self.return_named_point_cloud = return_named_point_cloud
        self.backend = backend

        self._check_D_is_zero()

        if self.backend == "torch" and "torch" not in sys.modules:
            raise RuntimeError(
                "Requested backend is torch, "
                "but torch module could not be imported.")
        if self.backend == "scipy":
            self.min_pool_fn = self._min_pool_with_scipy
        else:
            self.min_pool_fn = self._min_pool_with_torch

    def _check_D_is_zero(self):
        assert np.all(self.D == 0), "D != 0, but only rectified depth is supported"

    def _min_pool_with_scipy(self, input):
        pooled = minimum_filter(input, size=self.pool_size,
            origin=-(self.pool_size // 2), mode='constant', cval=np.inf,
            axes=(-2, -1))
        pooled = pooled[..., ::self.pool_size, ::self.pool_size]
        return pooled

    def _min_pool_with_torch(self, input):
        if input.ndim == 2:
            input = np.expand_dims(input, axis=0)
            expanded = True
        else:
            expanded = False
        max_pool_fn = torch.nn.MaxPool2d(self.pool_size)
        pooled = -max_pool_fn(-torch.from_numpy(input)).numpy()
        if expanded:
            pooled = pooled[0]
        return pooled

    def convert(self, depth):
        self._check_D_is_zero()

        if self.pool_size > 1:
            scale = get_depth_scale(depth)
            depth = depth * scale
            invalid = (depth <= 0) | ~np.isfinite(depth)
            depth[invalid] = np.inf
            depth = self.min_pool_fn(depth)
            K = self.K / self.pool_size
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            valid = np.isfinite(depth)
            z = depth[valid]
            v, u = np.where(valid)
            x = (u - cx) / fx * z
            y = (v - cy) / fy * z
        else:
            scale = get_depth_scale(depth)
            valid = (depth > 0) & np.isfinite(depth)
            fx = self.K[0, 0]
            fy = self.K[1, 1]
            cx = self.K[0, 2]
            cy = self.K[1, 2]
            z = depth[valid]
            z = z * scale
            v, u = np.where(valid)
            x = (u - cx) / fx * z
            y = (v - cy) / fy * z

        if self.return_named_point_cloud:
            point_cloud = np.rec.fromarrays([x, y, z], names=['x', 'y', 'z'])
        else:
            point_cloud = np.vstack((x, y, z)).transpose()

        return point_cloud
