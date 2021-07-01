import h5py
import numpy as np
import os
import pdb


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    points = np.squeeze(data, 0)
    return points


def write_off(ptsdata, filename):
    numV = ptsdata.shape[0]
    numF = 0
    numE = 0
    with open(filename, 'w+') as f:
        f.write('OFF\n{} {} {}\n'.format(numV, numF, numE))
    with open(filename, 'ab') as f:
        np.savetxt(f, ptsdata)


if __name__ == '__main__':
    h5fn = os.path.join('./points_data', 'Japanese_stool_5.hd5')
    points = load_h5(h5fn)
    pdb.set_trace()
    offname = os.path.join('./points_data', 'Japanese_stool_5.off')
    write_off(points, offname)

