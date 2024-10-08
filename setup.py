from setuptools import setup

setup(name = 'diffuser',
    version = '0.1',
    packages = ['diffuser'],
    install_requires = [
        'numpy',
        'h5py',
        'scipy',
        'scikit-optimize',
        'mrcfile',
        'mock', # Till MDAnalysis issue is fixed for python 3
        'MDAnalysis',
        'matplotlib',
    ],
    package_data = {'':['kernels.cu']
    },
    entry_points = {'console_scripts': [
        'diffuser.bgo_optimize = diffuser.bgo_optimize:main',
        'diffuser.bgo_tls = diffuser.bgo_tls:main',
        'diffuser.pcdiff = diffuser.pcdiff:main',
        'diffuser.tlsdiff = diffuser.tlsdiff:main',
        'diffuser.traj_diff = diffuser.traj_diff:main',
        'diffuser.rbdiff = diffuser.rbdiff:main',
        'diffuser.process_md = diffuser.process_md:main',
      ],
    },
)
