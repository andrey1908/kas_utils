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
            self.pool_fn = self._pool_depth_with_scipy
        else:
            self.pool_fn = self._pool_depth_with_torch

    def _check_D_is_zero(self):
        assert np.all(self.D == 0), "D != 0, but only rectified depth is supported"

    def _pool_depth_with_scipy(self, depth):
        pooled = minimum_filter(depth, size=self.pool_size,
            origin=-(self.pool_size // 2), mode='constant', cval=np.inf)
        pooled = pooled[::self.pool_size, ::self.pool_size]
        return pooled

    def _pool_depth_with_torch(self, depth):
        depth = np.expand_dims(depth, axis=0)
        max_pool_fn = torch.nn.MaxPool2d(self.pool_size)
        pooled = -max_pool_fn(-torch.from_numpy(depth)).numpy().squeeze()
        return pooled

    def convert(self, depth):
        self._check_D_is_zero()

        scale = get_depth_scale(depth)
        depth = depth * scale

        invalid = (depth <= 0) | ~np.isfinite(depth)
        depth[invalid] = np.inf

        pooled_depth = self.pool_fn(depth)
        pooled_K = self.K / self.pool_size
        fx = pooled_K[0, 0]
        fy = pooled_K[1, 1]
        cx = pooled_K[0, 2]
        cy = pooled_K[1, 2]

        valid = np.isfinite(pooled_depth)
        z = pooled_depth[valid]
        v, u = np.where(valid)
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z

        if self.return_named_point_cloud:
            point_cloud = np.rec.fromarrays([x, y, z], names=['x', 'y', 'z'])
        else:
            point_cloud = np.vstack((x, y, z)).transpose()

        return point_cloud
