# cython: language_level=3

from ocnn.virtualscanner import VirtualScanner
from ocnn.octree import Points
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from FarthestPointSampling.tf_sampling import FPS
import os
import glob
import pdb


data_root = './sphere'
out_root = './sphere'

if __name__ == '__main__':
    # converts obj/off file to points
    files = glob.glob(os.path.join(data_root, '*.obj'))
    snum = 4096 # 2048  # sampling num
    save_root = os.path.join(out_root, 'hdf5_'+str(snum))
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    for f in files:
        fn = f.split('/')[-1].split('.')[0]
        # fn = 'Children_bed_1'
        if not os.path.exists(os.path.join(out_root, fn+'.points')):
            scanner = VirtualScanner(filepath=os.path.join(data_root, fn+'.obj'),
                    view_num=6, flags=False, normalize=True)
            scanner.save(os.path.join(out_root, fn+'.points'))
        else:
            pass

        # read .points file and parse the point data
        points = Points(os.path.join(out_root, fn+'.points'))
        [pts, norms] = points.get_points_data()

        # Farthest Point Sampling
        save_fn = os.path.join(save_root, fn)
        print('sampling...')

        sampled_pts = FPS(pts, snum, save_fn)
        pts_new = np.squeeze(sampled_pts)

        # show 3D points
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(pts_new[:, 0], pts_new[:, 1], pts_new[:, 2], s=5)
        # plt.show()
        # pdb.set_trace()

        # save sampled points
