#!/usr/bin python

import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D

def fig_size(width, *args, fraction=1, subplot=[1,1], ratio=None):
    # https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy
    ratio: float
            ratio width / height

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """

    # pre-defined widths
    if width == 'article':
        width = 452.0
    if width == 'article_twocolumn':
        width = 452.0
    if width == 'article_twocolumn_cw':
        width = 221.0
    elif width == 'report':
        width = 360.0
    elif width == 'tex_template':
        width = 450.0
    elif width == 'ETHexercise':
        width = 345.0
    elif width == 'ETHexercise_wide':
        width = 455.0

    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if ratio is None:
        # Golden ratio to set aesthetic figure height
        hratio = (5**.5 - 1) / 2
    else:
        # ratio is specified as width/height, e.g. 16/9, 4/3 etc.
        hratio = 1/ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * hratio * (subplot[0] / subplot[1])

    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim

def font_conf(fontsize, subfontsize=None, *args, update=True):
    """Setup fontsizes for LaTeX integration without scaling"""
    if subfontsize is None:
        subfontsize = fontsize - 2 # 2pt smaller

    conf = {
        # LaTeX
        # "text.usetex": True,
        # "font.family": "serif",

        # font sizes
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "legend.fontsize": subfontsize,
        "xtick.labelsize": subfontsize,
        "ytick.labelsize": subfontsize,
        "figure.titlesize": fontsize
    }

    if update:
        mpl.rcParams.update(conf)

    return conf

def get_figure(nx=1, ny=1, *args,
               width='article_twocolumn', fontsize=12,
               ratio=None, fraction=1.0, **kwargs):
    """Get a set of figure and axes handles with default style and configuration
    set up properly (convenience function)"""

    # FIXME: [fabianw@mavt.ethz.ch; 2019-06-02] this can not be done in the
    # matplotlibrc/mplstyle file directly
    curr_cycler = mpl.rcParams['axes.prop_cycle']
    linestyles = cycler.cycler('linestyle',
            ('-', '--', '-.', ':',             # user defined default
                (0,(3.0,3.8)),                 # loosely dashed
                (0,(4.2,2.5,1,1.8,1,2.5)),     # dash-dot-dot
                (0,(1,1.5)),                   # densely dotted
                (0,(4.2,2.5,1,1.8,1,1.8,1,2.5) # dash-dot-dot-dot
                    )
                )
            )
    if len(curr_cycler) <= len(linestyles):
        # add linestyles if curr_cycler is of same length (8 styles by default)
        mpl.rcParams['axes.prop_cycle'] = curr_cycler + linestyles[:len(curr_cycler)]
    font_conf(fontsize)
    figsize = fig_size(width, subplot=[nx,ny], ratio=ratio, fraction=fraction)
    fig, ax = plt.subplots(nx, ny, figsize=figsize, **kwargs)

    return fig, ax

def get_cycler():
    """Return individual color, linestyle and marker cyclers depending on
    mpl.rcParams"""
    N = len(mpl.rcParams['axes.prop_cycle']) # number of elements in cycler
    kcycler = mpl.rcParams['axes.prop_cycle'].by_key()
    split = {}
    keys = ['color', 'c', 'linestyle', 'ls', 'marker']
    for k in keys:
        if k in kcycler:
            split[k] = cycler.cycler(k, kcycler[k])
    if 'marker' not in split:
        markers = list('os^dv8<p>hXDPHx+1234')
        assert N <= len(markers)
        split['marker'] = cycler.cycler('marker', markers[:N])
    return split

def export_keys(lcycler, mcycler, ccycler=None, *args,
        fontsize=11, handlelength=3, margins=[1.0, 0.0], outdir='./keys'):
    """Export legend key entries into chopped up pdf pieces, to be used in LaTeX
    captions etc."""
    # lcycler: linestyle cycler
    # mcycler: marker cycler
    # ccycler: color cycler (if None black color is used)

    import re, shlex, glob, os
    import inspect
    import subprocess as sp

    assert len(lcycler) == len(mcycler)
    if ccycler is not None:
        assert len(ccycler) == len(lcycler)
    kcycler = cycler.cycler(color=['k' for x in range(len(lcycler))]) # black
    lmcycler = lcycler + mcycler # linestyle and marker
    # disable linestyle for marker only
    nolcycler = cycler.cycler(linestyle=['none' for x in range(len(mcycler))])
    mcycler += nolcycler

    prefix = '012345tmp'

    # determine font height from fontsize and compute bounding box height dy
    minimal = inspect.cleandoc(r"""
    \documentclass[a4paper, {fontsize}pt]{{article}}
    \begin{{document}}
    \newlength{{\fheight}}
    \settoheight{{\fheight}}{{f}}
    \showthe\fheight
    \end{{document}}
    """.format(fontsize = int(fontsize)))
    fname = prefix + '.tex'
    with open(fname, 'w') as out:
        out.write(minimal)
    cmd = "pdflatex -interaction nonstopmode {fname:s}".format(fname=fname)
    out = sp.run(shlex.split(cmd), capture_output=True).stdout.decode('utf-8')
    dy = float(re.search(r'^>\s*([0-9\.]*)pt\.$', out, re.MULTILINE).group(1))
    for trash in glob.glob(prefix + '*'): # clean up
        os.remove(trash)

    # export keys to outdir (requires pdfcrop)
    predir = os.path.join(outdir, 'preview')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    if not os.path.isdir(predir):
        os.mkdir(predir)

    colors = {'k' : kcycler}
    if ccycler is not None:
        colors['c'] = ccycler
    styles = {
            'l'  : lcycler,
            'm'  : mcycler,
            'lm' : lmcycler
            }

    ftmp = prefix + 'key.pdf'

    with open(os.path.join(predir, 'preview.tex'), 'w') as preview:
        texbuf = inspect.cleandoc(r"""
        \documentclass[a4paper, {fontsize}pt]{{article}}
        \usepackage{{graphicx}}
        \newcommand{{\gkey}}[1]{{\protect\includegraphics{{#1.pdf}}}}
        \graphicspath{{{{../}}}}
        \begin{{document}}
        \begin{{itemize}}
        """.format(fontsize = int(fontsize)))
        preview.write(texbuf+'\n')
        for c,cc in colors.items():
            for s,sc in styles.items():
                all_cycler = cc + sc # combine all into one cycler
                for i,style in enumerate(all_cycler):
                    key = '{id:02d}{cid:s}{sid:s}'.format(id=i+1, cid=c, sid=s)
                    dst = os.path.join(outdir, key+'.pdf')
                    preview.write("\\item[\\texttt{{{key:s}}}:] fff \\gkey{{{key:s}}}~Xxx\n".format(
                        key = key))
                    # generate key
                    h = mlines.Line2D([], [], **style)
                    plt.legend(handles=[h],
                            fontsize=fontsize,
                            handlelength=handlelength,
                            loc='lower left', bbox_to_anchor=[0,0])
                    plt.gca().set_axis_off()
                    plt.savefig(ftmp, bbox_inches='tight')

                    # compute hires bounding box via pdfcrop
                    cmd = "pdfcrop --hires --verbose {fname:s} /dev/null".format(fname=ftmp)
                    out = sp.check_output(shlex.split(cmd)).decode('utf-8')
                    bbox = re.search(r'^\s*%%HiResBoundingBox:\s*([0-9\s\.]*)$',
                            out, re.MULTILINE).group(1)
                    bbox = [float(x) for x in bbox.split()]

                    # crop and save
                    cx = bbox[0]
                    dx = bbox[2] - cx
                    cy = 0.5 * (bbox[1] + bbox[3] - dy)
                    cmd = "pdfcrop --hires --bbox '{xs:f} {ys:f} {xe:f} {ye:f}' --margins '{mlr:f} {mtb:f}' {fname:s} {dst:s}".format(
                            xs = cx,
                            ys = cy,
                            xe = cx + dx,
                            ye = cy + dy,
                            mlr = margins[0], # margins left/right
                            mtb = margins[1], # margins top/bottom
                            fname = ftmp,
                            dst = dst
                            )
                    sp.run(shlex.split(cmd))
        preview.write("\end{itemize}\n\end{document}")
    os.remove(ftmp) # clean up

def latex_textwidth(class_name, fontsize='11pt'):
    """Create a minimal LaTeX document to determine the textwidth in pt used for
    the given class"""

    import inspect

    minimal = inspect.cleandoc(r"""
    \documentclass[a4paper, {fontsize:s}]{{{class_name:s}}}
    \begin{{document}}
    % gives the width of the current document in pts (full)
    % for 2-columns: \showthe\columnwidth
    \showthe\textwidth
    \showthe\columnwidth
    \end{{document}}
    """.format(class_name = class_name, fontsize = fontsize))

    with open('latexTextwidth.tex', 'w') as out:
        out.write(minimal)
