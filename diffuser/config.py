import os.path as op
import configparser
import numpy as np
import h5py

_UNSET = object()

class DiffuserConfig(configparser.ConfigParser):
    '''Custom ConfigParser with utility functions'''
    def __init__(self, config_file):
        super().__init__()
        self.read(config_file)
        self.config_folder = op.dirname(config_file)

    def get_path(self, section, key, *, fallback=_UNSET):
        '''Get path to file assuming they are relative to config file'''
        if fallback is _UNSET:
            param = self.get(section, key)
        else:
            param = self.get(section, key, fallback=fallback)

        if param is None:
            return None
        return op.join(self.config_folder, param)

    def get_size(self):
        '''Return 3-vector of size parameters'''
        size = [int(s) for s in self.get('parameters', 'size').split()]

        if len(size) == 1:
            size = np.array(size * 3)
        elif len(size) != 3:
            raise ValueError('size parameter must be either 1 or 3 space-separated numbers')
        else:
            size = np.array(size)

        return size

    def get_qvox(self, size=None):
        '''Get qvox matrix from file or res_edge

        If qvox file not present, then the size parameter is required
        '''
        qvox_fname = self.get_path('files', 'qvox_fname', fallback=None)
        res_edge_str = self.get('parameters', 'res_edge', fallback='0.')
        res_edge = [float(r) for r in res_edge_str.split()]

        # Get voxel size in 3D
        if qvox_fname is not None and res_edge != [0.]:
            raise ValueError('Both res_edge and qvox_fname defined. Pick one.')
        if qvox_fname is not None:
            with h5py.File(qvox_fname, 'r') as fptr:
                qvox = fptr['q_voxel_size'][:]
        elif res_edge != [0.]:
            if len(res_edge) == 1 and res_edge[0] != 0.:
                res_edge = np.array(res_edge * 3)
            elif len(res_edge) != 3:
                raise ValueError('res_edge parameter must be either 1 or 3 space-separated numbers')
            else:
                res_edge = np.array(res_edge)

            if size is None:
                raise ValueError('Need size array if calculating qvox from res_edge')
            qvox = np.diag(1. / res_edge / (size//2))
        else:
            raise ValueError('Need either res_edge of qvox_fname to define voxel parameters')

        return qvox

    def get_traj_list(self):
        '''Get trajectory file list suitable for MDAnalysis universe'''
        traj_fname = self.get_path('files', 'traj_fname', fallback=None)
        traj_flist = self.get_path('files', 'traj_flist', fallback=None)

        if traj_fname is not None and traj_flist is not None:
            raise AttributeError('Cannot specify both traj_fname and traj_flist. Pick one.')
        if traj_fname is not None:
            return [traj_fname]
        if traj_flist is not None:
            with open(traj_flist, 'r') as fptr:
                flist = [l.strip() for l in fptr.readlines()]
            return flist
        return None

    def get_bounds(self, section, key, fallback='0 0'):
        '''Get min/max bounds from two whitespace separated numbers'''
        return tuple([float(s) for s in self.get(section, key, fallback=fallback).split()])
