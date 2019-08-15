import sys
import os.path as op
import numpy as np
import h5py

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

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
                self.cvol = f.data
        elif ext == '.h5':
            with h5py.File(fname, 'r') as f:
                self.cvol = f[self.dset][:]
                if len(self.cvol.shape) == 4:
                    self.cvol = self.cvol[0]
        else:
            raise IOError('Unknown file extension: %s'%ext)

        self.min, self.max = self.cvol.min(), self.cvol.max()

    def _gen_mesh(self, level):
        verts, faces = pg.isosurface(self.cvol, level)
        print(verts[:,0].min(), verts[:,0].max())
        self.mdata = gl.MeshData(verts, faces)

        colors = np.zeros((self.mdata.faceCount(), 4), dtype='f8')
        #colors[:,0] = (verts[faces[:,0], 0] - verts[:,0].min()) / (verts[:,0].max() - verts[:,0].min())
        colors[:,0] = np.linspace(0, 1, colors.shape[0])
        colors[:,2] = 0.5
        colors[:,3] = 0.7 # Opacity
        self.mdata.setFaceColors(colors)

    def _init_ui(self):
        self.resize(800,800)
        layout = QtWidgets.QVBoxLayout()
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        widget.setLayout(layout)

        # 3D GLView
        w = gl.GLViewWidget()
        layout.addWidget(w)
        w.setCameraPosition(distance=200)

        #g = gl.GLAxisItem()
        #g.setSize(100,100,100)
        #w.addItem(g)

        # -- Mesh Item
        self.mitem = gl.GLMeshItem(meshdata=self.mdata, smooth=True, shader='balloon', glOptions='additive')
        #mitem = gl.GLMeshItem(meshdata=self.mdata, smooth=True, shader='shaded', glOptions='additive')
        shift = -np.array(self.cvol.shape)/2
        self.mitem.translate(*tuple(shift))
        w.addItem(self.mitem)

        # Level slider
        line = QtWidgets.QHBoxLayout()
        layout.addLayout(line)
        self.level_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.level_slider.setRange(0, self.num_levels)
        self.level_slider.setValue(int(0.7*self.num_levels))
        self.level_slider.sliderMoved.connect(self._level_changed)
        self.level_slider.sliderReleased.connect(self._level_updated)
        line.addWidget(self.level_slider)
        self.level_value = QtWidgets.QLineEdit(str(self.min), self)
        self.level_value.setFixedWidth(80)
        self.level_value.editingFinished.connect(self._level_updated)
        self.level_value.setText('%.2e'%self._level_from_value(int(0.7*self.num_levels)))
        line.addWidget(self.level_value)

        self.show()

    def _level_updated(self):
        value = float(self.level_slider.value())
        level = self.min + value/self.num_levels * (self.max - self.min)
        self._gen_mesh(level)
        self.mitem.setMeshData(meshdata=self.mdata)

    def _level_changed(self, value=None):
        if value is None:
            level = float(self.level_value.text())
            value = self._value_from_level(level)
            self.level_slider.setValue(value)
        else:
            level = self._level_from_value(value)
            self.level_value.setText('%.2e'%level)

    def _value_from_level(self, level):
        return int(np.round(float(self.num_levels) * (level - self.min) / (self.max - self.min)))

    def _level_from_value(self, value):
        return self.min + float(value)/self.num_levels * (self.max - self.min)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='3D Isosurface viewer')
    parser.add_argument('file', help='Path to 3D volume')
    parser.add_argument('-d', '--dset', help='HDF5 dataset if applicable (Default: /intens)', default='/intens')
    args = parser.parse_args()
    
    app = QtGui.QApplication([])
    #v = Viewer3D('/home/ayyerkar/acads/ATP/2w5j_cutout_2mFo-DFc.omap')
    v = Viewer3D(args.file, args.dset)
    sys.exit(app.exec_())
