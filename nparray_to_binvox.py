import numpy as np
import os
import sys
class Voxels(object):
    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order

    def clone(self):
        data = self.data.copy()
        dims = self.dims[:]
        translate = self.translate[:]
        return Voxels(data, dims, translate, self.scale, self.axis_order)

    def write(self, fp):
        write(self, fp)

def sparse_to_dense(voxel_data, dims, dtype=np.bool):
    if voxel_data.ndim!=2 or voxel_data.shape[0]!=3:
        raise ValueError('voxel_data is wrong shape; should be 3xN array.')
    if np.isscalar(dims):
        dims = [dims]*3
    dims = np.atleast_2d(dims).T
    # truncate to integers
    xyz = voxel_data.astype(np.int)
    # discard voxels that fall outside dims
    valid_ix = ~np.any((xyz < 0) | (xyz >= dims), 0)
    xyz = xyz[:,valid_ix]
    out = np.zeros(dims.flatten(), dtype=dtype)
    out[tuple(xyz)] = True
    return out

def write(voxel_model, fp):
    """ 
    Write binary binvox format.

    Note that when saving a model in sparse (coordinate) format, it is first
    converted to dense forma    t.

    Doesn't check if the model is 'sane'.

    """
    if voxel_model.data.ndim==2:
        # TODO avoid conversion to dense
        dense_voxel_data = sparse_to_dense(voxel_model.data, voxel_model.dims)
    else:
        dense_voxel_data = voxel_model.data

    fp.write('#binvox 1\n'.encode('ascii'))
    line = 'dim '+' '.join(map(str, voxel_model.dims))+'\n'
    fp.write(line.encode('ascii'))
    line = 'translate '+' '.join(map(str, voxel_model.translate))+'\n'
    fp.write(line.encode('ascii'))
    line = 'scale '+str(voxel_model.scale)+'\n'
    fp.write(line.encode('ascii'))
    fp.write('data\n'.encode('ascii'))
    if not voxel_model.axis_order in ('xzy', 'xyz'):
        raise ValueError('Unsupported voxel model axis order')

    if voxel_model.axis_order=='xzy':
        voxels_flat = dense_voxel_data.flatten()
    elif voxel_model.axis_order=='xyz':
        voxels_flat = np.transpose(dense_voxel_data, (0, 2, 1)).flatten()

    # keep a sort of state machine for writing run length encoding
    state = voxels_flat[0]
    ctr = 0
    for c in voxels_flat:
        if c==state:
            ctr += 1
            # if ctr hits max, dump
            if ctr==255:
                fp.write(state.tobytes())
                fp.write(ctr.to_bytes(1, byteorder='little'))
                ctr = 0
        else:
            # if switch state, dump
            if ctr > 0:
                fp.write(state.tobytes())
                fp.write(ctr.to_bytes(1, byteorder='little'))
            state = c
            ctr = 1
    # flush out remainders
    if ctr > 0:
        fp.write(state.tobytes())
        fp.write(ctr.to_bytes(1, byteorder='little'))


def main(argv):
    for arg in argv[1:]:
        data = np.load(arg).astype(np.bool)
        dims = data.shape
        translate = np.zeros(3)
        scale = 41.133
        axis_order = 'xyz'
        voxel = Voxels(data, data.shape, translate, scale, axis_order)
        newname=[]
        fileName = os.path.splitext(arg)[0]
        newname.append(fileName)
        new = str(newname)+'.binvox'
        with open(new, 'wb') as f:
            voxel.write(f)
    pass


if __name__ == '__main__':
    main(sys.argv)



