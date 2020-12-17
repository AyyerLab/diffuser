import sys
import os.path as op
import numpy as np
import h5py

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from skimage import measure

import mrcfile
import parse_dsn6

class Viewer3D(QtWidgets.QMainWindow):
    def __init__(self, fname, dset='/intens'):
        super(Viewer3D, self).__init__()
        self.dset = dset
        self.num_levels = 1000
        self._parse(fname)
        self._gen_mesh(self.min + 0.7*(self.max-self.min))
        self._init_ui()

    def _parse(self, fname):
        ext = op.splitext(fname)[1]
        if ext == '.dsn6' or ext == '.omap':
            self.cvol, self.ccell = parse_dsn6.parse(fname)
            if self.cvol[0].mean() > self.cvol[self.cvol.shape[0]//2].mean():
                self.cvol = np.fft.fftshift(self.cvol)
        elif ext == '.ccp4' or ext == '.mrc':
            with mrcfile.open(fname, 'r') as f:
                self.cvol = np.copy(f.data)
        elif ext == '.h5':
            with h5py.File(fname, 'r') as f:
                self.cvol = f[self.dset][:]
                if len(self.cvol.shape) == 4:
                    self.cvol = self.cvol[0]
        else:
            raise IOError('Unknown file extension: %s'%ext)

        self.min, self.max = self.cvol.min(), self.cvol.max()

    def _init_ui(self):
        self.resize(800,800)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.setCentralWidget(splitter)

        # 3D GLView
        self.w3d = gl.GLViewWidget()
        splitter.addWidget(self.w3d)
        self.w3d.setCameraPosition(distance=200)

        # -- Mesh Item
        self.mitem = gl.GLMeshItem(meshdata=self.mdata, smooth=True, shader='balloon', glOptions='additive')
        #mitem = gl.GLMeshItem(meshdata=self.mdata, smooth=True, shader='shaded', glOptions='additive')
        shift = -np.array(self.cvol.shape)/2
        self.mitem.translate(*tuple(shift))
        self.w3d.addItem(self.mitem)

        # Histogram
        self.levelplot = pg.PlotWidget()
        splitter.addWidget(self.levelplot)
        self.levelplot.setLogMode(y=True)
        self.level = pg.InfiniteLine(0, movable=True)
        self.level.sigPositionChangeFinished.connect(self._level_updated)
        self.levelplot.addItem(self.level)
        self._refresh_levelplot()

        splitter.setSizes([400,100])
        self.show()

    def _gen_mesh(self, level):
        verts, faces = measure.marching_cubes_lewiner(self.cvol, level)[:2]
        #verts, faces = pg.isosurface(self.cvol, level)
        #print(verts[:,0].min(), verts[:,0].max())
        self.mdata = gl.MeshData(verts, faces)

        colors = np.ones((self.mdata.faceCount(), 4), dtype='f8') * 0.2
        cen = np.array(self.cvol.shape) // 2
        #colors[:,0] = (verts[faces[:,0], 0] - verts[:,0].min()) / (verts[:,0].max() - verts[:,0].min())
        #colors[:,0] = np.linspace(0, 1, colors.shape[0])
        colors[:,2] = np.linalg.norm((verts[faces[:,0]] - cen) / cen, axis=1)
        #colors[:,2] = 1
        colors[:,3] = 0.7 # Opacity
        self.mdata.setFaceColors(colors)

    def _refresh_levelplot(self):
        hy, hx = np.histogram(self.cvol.ravel(), bins=500)
        hy[hy==0] = 1
        self.levelplot.plot(hx, hy, stepMode=True, fillLevel=0.1, brush=(51,51,255,51))
        vmin, vmax = self.cvol.min(), self.cvol.max()
        self.level.setPos(vmin + 0.7*(vmax-vmin))

    def _level_updated(self):
        self._gen_mesh(self.level.value())
        self.mitem.setMeshData(meshdata=self.mdata)
        self.mitem.meshDataChanged()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='3D Isosurface viewer')
    parser.add_argument('file', help='Path to 3D volume')
    parser.add_argument('-d', '--dset', help='HDF5 dataset if applicable (Default: /intens)', default='/intens')
    args = parser.parse_args()
    
    app = QtGui.QApplication([])
    v = Viewer3D(args.file, args.dset)
    sys.exit(app.exec_())
