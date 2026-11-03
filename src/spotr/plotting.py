"""Spatial plots through squidpy: spots() draws one value or colour per spot, cells() draws single cells as points.

`colors` is a list of colours (one per spot or cell, drawn as a categorical whose categories are the colours themselves, so callers keep their
own palettes) or a numeric array (drawn with `cmap`).  Spots are squares of side `pitch` or Visium circles (shape="circle"); with `image` (an
RGB array and its scale factors, as Space Ranger writes them) the tissue image is drawn underneath and `xy` are full-resolution pixel
coordinates.
"""
import numpy as np


def _adata_for(xy, values, key="value", pitch=None, image=None, library="slide"):
    import anndata as ad, pandas as pd
    a = ad.AnnData(np.zeros((len(xy), 1), np.float32)); a.obsm["spatial"] = np.asarray(xy, float)
    if isinstance(values, (list, tuple)) or (isinstance(values, np.ndarray) and values.dtype.kind in "OUS"):
        from matplotlib.colors import to_hex
        hx = [to_hex(v, keep_alpha=True) for v in values]; cats = list(dict.fromkeys(hx)); a.obs[key] = pd.Categorical(hx, categories=cats); a.uns[f"{key}_colors"] = cats
    else:
        a.obs[key] = np.asarray(values, float)
    if image is not None:
        a.uns["spatial"] = {library: dict(images=dict(lowres=image["img"]), scalefactors=dict(image["scalefactors"]))}
    elif pitch is not None:
        a.uns["spatial"] = {library: dict(images={}, scalefactors=dict(spot_diameter_fullres=float(pitch), tissue_hires_scalef=1.0, tissue_lowres_scalef=1.0))}
    return a


def cells(ax, xy, colors, title, size=4.0, fontsize=10, extent=None):
    """single cells as points (y down, as spots() draws spots); `extent` (xmin, xmax, ymin, ymax) to match a spots() panel"""
    import squidpy as sq
    a = _adata_for(xy, colors)
    sq.pl.spatial_scatter(a, shape=None, color="value", ax=ax, title=title, legend_loc=None, frameon=False, size=size)
    for c_ in ax.collections:                                                                   # thousands of points: rasterized, so the PDF loads fast
        c_.set_rasterized(True)
    ax.set_aspect("equal"); ax.set_xlabel(""); ax.set_ylabel(""); ax.title.set_fontsize(fontsize)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[3], extent[2])
    return ax


def spots(ax, xy, colors, title, pitch=None, image=None, cmap=None, norm=None, shape="square", flip=False, scalebar_um=None, fontsize=10, size=1.0, img_alpha=None, colorbar=None):
    """one value per spot on `ax`; squares of side `pitch` (default: the median nearest-neighbour distance) unless `image` is given; with `image`,
    `size` scales the Visium spot diameter and `img_alpha` dims the tissue image under the spots"""
    import squidpy as sq
    xy = np.asarray(xy, float).copy()
    if flip:
        xy[:, 1] = -xy[:, 1]
    if pitch is None and image is None:
        from scipy.spatial import cKDTree
        pitch = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    if colors is None and image is not None:                                                         # the tissue image alone
        a = _adata_for(xy, np.zeros(len(xy)), pitch=pitch, image=image); sq.pl.spatial_scatter(a, library_id="slide", img_res_key="lowres", color=None, ax=ax, title=title, frameon=False)
        ax.set_aspect("equal"); ax.set_xlabel(""); ax.set_ylabel(""); ax.title.set_fontsize(fontsize); return ax
    a = _adata_for(xy, colors, pitch=pitch, image=image)
    kw = dict(color="value", ax=ax, title=title, legend_loc=None, colorbar=(cmap is not None) if colorbar is None else colorbar, frameon=False)   # colorbar=False: the caller places it
    if image is not None:
        sq.pl.spatial_scatter(a, library_id="slide", img_res_key="lowres", shape="circle", size=size, img_alpha=img_alpha, cmap=cmap, norm=norm, na_color=(0, 0, 0, 0), **kw)
    else:
        sq.pl.spatial_scatter(a, library_id="slide", img=False, shape=shape, scale_factor=1.0, size=1.0, cmap=cmap, norm=norm, **kw)
    for c_ in ax.collections:                                                                   # thousands of spots: rasterized, so the PDF loads fast
        c_.set_rasterized(True)
    ax.set_aspect("equal"); ax.set_xlabel(""); ax.set_ylabel(""); ax.title.set_fontsize(fontsize)
    if scalebar_um and image is None:
        lo, hi = xy.min(0), xy.max(0); y0 = (lo[1] - 0.6 * pitch) if not ax.yaxis_inverted() else (hi[1] + 0.6 * pitch)
        ax.plot([hi[0] - scalebar_um, hi[0]], [y0, y0], color="#111111", lw=2, solid_capstyle="butt", clip_on=False)
        ax.text(hi[0] - scalebar_um / 2, y0, f"{scalebar_um:g} µm", ha="center", va="top" if not ax.yaxis_inverted() else "bottom", fontsize=7)
    return ax
