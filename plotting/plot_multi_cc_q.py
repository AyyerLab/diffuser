import numpy as np
import matplotlib
import matplotlib.pylab as P
from matplotlib.pyplot import cm


def multi_cc_q(*args, **kwargs):
    ax = kwargs.get("ax", None)
    q = kwargs.get("q", None)
    cc = kwargs.get("cc", None)

    if ax is None:
        ax = P.gca()

    ax.set_title("CC v/s q")
    ax.set_xlabel("q [$\AA$]")
    ax.set_ylabel("CC")
    ax.grid()

    if cc is not None:
        ax.set_cc(cc)

    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    for fname, color in zip(args, colors):
        if isinstance(fname, tuple):
            name, fname = fname
        else:
            name = None

        corr = np.loadtxt(fname, skiprows =1)
        q = corr[:, 1]
        cc = corr[:, 2]
        #ax=P.axes(xlim =(0, 1.4), ylim =(0, 1.0))
        ax.plot(q, cc, c=color, lw=2, label = name)
        P.legend()
    P.show()
    return ax    

multi_cc_q(('5','/home/mazumdep/CypA/xtc/md295_forest_md_5vecs_diag_1_diffcal_CC.dat'), ('20', '/home/mazumdep/CypA/xtc/md295_forest_md_20vecs_diag_diffcalc_CC.dat'))
