"""Panel layout for the multi-panel figures of the paper, with the panels flush with the edges of the figure and the letters on the edges of the
panels.

Spatial panels keep an equal aspect, so matplotlib shrinks them inside their grid cell and centers them, which leaves them inset from the edge of
the figure while the plots beside or below them fill their cells. `finish(fig, panels)` therefore

  1. pins every equal-aspect panel to the top of its cell, and to the left of it (to the right in the last column of the figure),
  2. groups the panels by the left edge of their grid cell (a column) and shifts each panel so that its left extent, which is the axes or its y
  tick labels and label when it has them, starts at the leftmost extent of the column,
  3. puts each letter (bold, uppercase) at the left edge of its column, just above the highest title of its row.

Call it once, after everything is drawn and before saving.
"""
import numpy as np


def _left_top(ax, r, inv):
    """Return the left extent (axes or y-axis labels) and the top (axes or title) of the panel, in figure coordinates."""
    bb = [ax.get_window_extent(r)]
    if ax.axison and ax.yaxis.get_visible():
        t = ax.yaxis.get_tightbbox(r)
        if t is not None and t.width > 0:
            bb.append(t)
    x0 = min(inv.transform((b.x0, b.y0))[0] for b in bb)
    top = inv.transform((0, ax.get_window_extent(r).y1))[1]
    if ax.title.get_text():
        top = max(top, inv.transform((0, ax.title.get_window_extent(r).y1))[1])
    return x0, top


def finish(fig, panels, fontsize=15, tol=0.01, pad=0.006):
    """Align the panels and place their letters, with `panels` as [(ax, "A"), ...] in reading order."""
    if not panels:
        return
    cells = {id(ax): ax.get_subplotspec().get_position(fig) if ax.get_subplotspec() is not None else ax.get_position() for ax, _ in panels}
    right = max(c.x1 for c in cells.values())
    for ax, _ in panels:
        if ax.get_aspect() != "auto":
            ax.set_anchor("NE" if abs(cells[id(ax)].x1 - right) < tol and abs(cells[id(ax)].x0 - min(c.x0 for c in cells.values())) > tol else "NW")
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    # columns, the panels with the same left edge of the grid cell
    cols = []
    for ax, _ in panels:
        x = cells[id(ax)].x0
        for c in cols:
            if abs(c[0] - x) < tol:
                c[1].append(ax); break
        else:
            cols.append([x, [ax]])
    for _, axs in cols:
        ext = {id(a): _left_top(a, r, inv)[0] for a in axs}; target = min(ext.values())
        for a in axs:
            d = target - ext[id(a)]
            if abs(d) > 1e-4:
                p = a.get_position(); a.set_position([p.x0 + d, p.y0, p.width, p.height])
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    lt = {id(ax): _left_top(ax, r, inv) for ax, _ in panels}
    colx = {}
    for _, axs in cols:
        x = min(lt[id(a)][0] for a in axs)
        for a in axs:
            colx[id(a)] = x
    # rows, the panels with the same top of the grid cell
    rows = []
    for ax, _ in panels:
        y = cells[id(ax)].y1
        for rw in rows:
            if abs(rw[0] - y) < tol:
                rw[1].append(ax); break
        else:
            rows.append([y, [ax]])
    rowy = {}
    for _, axs in rows:
        y = max(lt[id(a)][1] for a in axs)
        for a in axs:
            rowy[id(a)] = y
    for ax, s in panels:
        fig.text(colx[id(ax)], rowy[id(ax)] + pad, s, fontsize=fontsize, fontweight="bold", ha="left", va="bottom", color="#111111")


def compress_pdf(path, dpi=250):
    """Recompress the raster images of a saved figure (H&E, rasterized cells and spots) as JPEG at `dpi` with ghostscript, keeping vectors and text,
    so that the PDFs of the paper load quickly. Does nothing when ghostscript is not installed.
    """
    import shutil, subprocess, tempfile, os
    if shutil.which("gs") is None:
        return
    fd, tmp = tempfile.mkstemp(suffix=".pdf"); os.close(fd)
    r = subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5", "-dPDFSETTINGS=/printer",
                        f"-dColorImageResolution={dpi}", f"-dGrayImageResolution={dpi}", f"-sOutputFile={tmp}", str(path)])
    if r.returncode == 0 and os.path.getsize(tmp) < os.path.getsize(path):
        shutil.move(tmp, path)
    elif os.path.exists(tmp):
        os.remove(tmp)
