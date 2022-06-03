import os
from datetime import datetime
import numpy as np
from pathlib import Path
from matplotlib import path
from netCDF4 import Dataset
from . import physical
from . import matrix
from . import dataset


def ismember(a, b):
    """
    Re-implementation of ismember() from Matlab but return only the second arg which is the index
    https://stackoverflow.com/questions/15864082/python-equivalent-of-matlabs-ismember-function
    Args:
        a: first list
        b: second list to compare

    Returns:
        list size of a: value is the indices in the list b (-1 if not present)
        ex: ismember([1,2,4], [1,2,3,5]) = [0, 1, -1]
    """
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return np.array([bind.get(itm, -1) for itm in a])  # -1 if not in b


def filter_vector(vector, keep):
    """
    Remove element from vector (or vectors is a list) according to a index list or boolean vector
    vectors: Numpy array or list of Numpy array
    Args:
        vector: array or list of arrays to filter
        keep: list of index to keep or a boolean array with the same size as vector

    Returns:
        outlist: filtered array or list of arrays
    """
    if type(vector) != list:
        return vector[keep]
    else:
        outlist = []
        for vec in vector:
            outlist.append(vec[keep])
        return outlist


def bins_in_contour(d, xc, yc, return_path=False):
    """
    Retrieve bins located inside a contour defined by the lists xc and yc

    Args:
        xc: latitude of the contour
        yc: longitude of the contour
        d: physical domain object
    Return:
        p: path from the (xc, yc) coordinates
        bins: list of bins inside the contour
    """
    p = path.Path(np.vstack((xc, yc)).T)
    bins_xy = np.where(p.contains_points(d.coords))[0]
    bins = np.where(
        np.any(np.in1d(d.bins.reshape(1, -1), bins_xy).reshape(len(d.bins), 4), 1)
    )[0]
    return (np.unique(bins), p) if return_path else np.unique(bins)


def segments_in_contour(data, xc, yc, segments=None):
    """
    Retrieve trajectory segments located inside a contour defined by (xc, yc)

    Args:
        xc: latitude of the contour
        yc: longitude of the contour
        data: dataset object (contains x0, xt, y0, yt)
        segments ['start', 'end', None]: consider either the start, the end, or both positions of each segments
    Return:
        logical arrays len(data.x0)
    """
    p = path.Path(np.vstack((xc, yc)).T)

    # considering begining (x0), end (xt) or both
    if segments == "start":
        s_in = p.contains_points(np.vstack((data.x0, data.y0)).T)
    elif segments == "end":
        s_in = p.contains_points(np.vstack((data.xt, data.yt)).T)
    else:
        s_in = np.logical_or(
            p.contains_points(np.vstack((data.x0, data.y0)).T),
            p.contains_points(np.vstack((data.xt, data.yt)).T),
        )
    return s_in


def filter_region(data, xc, yc):
    """
    Remove segments inside a region defined by the contour coordinates (x,y)
    Args:
        xc: latitude of the contour
        yc: longitude of the contour
        data: dataset object (contains x0, xt, y0, yt)
        segments ['start', 'end', None]: consider either the start, the end, or both positions of each segments
    Return:
        logical arrays len(data.x0)
    """
    rpath = path.Path(np.vstack((xc, yc)).T)
    keep = np.where(~rpath.contains_points(np.vstack((data.x0, data.y0)).T))[0]
    [data.x0, data.y0, data.xt, data.yt] = filter_vector(
        [data.x0, data.y0, data.xt, data.yt], keep
    )


def remove_communication(d, data, x1, y1, x2, y2):
    """
    Remove the connection between two regions defined by (x1, y1) and (x2, y2)
    The two regions have to share a common edge, e.g. two regions: Atlantic and
    Pacific oceans with a common intersection at the Isthmus of Panama

    Args:
        d: physical domain object
        data: dataset object (contains x0, xt, y0, yt)
        (x1, y1): close contour defining first region
        (x2, y2): close contour defining second region
    """
    # bins first and second region
    bins_r1 = bins_in_contour(d, x1, y1)
    bins_r2 = bins_in_contour(d, x2, y2)

    # get unique and the intersection of both region
    bins_inter = np.intersect1d(bins_r1, bins_r2)

    # identify the segments from the data object that are in both region
    s1 = segments_in_contour(data, x1, y1)
    s2 = segments_in_contour(data, x2, y2)

    # for each bin we calculate how much segments are in r1 and r2
    remove = np.empty(0)
    bins_xy0 = d.find_element(data.x0, data.y0)
    bins_xyt = d.find_element(data.xt, data.yt)

    for b_i in bins_inter:
        s1i = np.logical_and(np.logical_or(bins_xy0 == b_i, bins_xyt == b_i), s1)
        s2i = np.logical_and(np.logical_or(bins_xy0 == b_i, bins_xyt == b_i), s2)

        if np.sum(s1i) < np.sum(s2i):
            remove = np.append(remove, np.arange(len(data.x0))[s1i])
        else:
            remove = np.append(remove, np.arange(len(data.x0))[s2i])

    # remove once at the end
    keep = np.setdiff1d(np.arange(0, len(data.x0)), remove)
    [data.x0, data.y0, data.xt, data.yt] = filter_vector(
        [data.x0, data.y0, data.xt, data.yt], keep
    )


def remove_panama_communication(d, data):
    """
    Function that call remove_communication_two_regions for a specific case
    with predefined region boundaries
    """
    # Isthmus of Panama
    x_po = np.array([-105, -105, -100.5, -85.5, -83, -81, -79.5, -77.5, -75, -70, -105])
    y_po = np.array([0, 25, 20, 13, 9, 8.25, 9.25, 8.5, 6, 0, 0])
    x_ao = np.array([-70, -105, -100.5, -85.5, -83, -81, -79.5, -77.5, -75, -70, -70])
    y_ao = np.array([25, 25, 20, 13, 9, 8.25, 9.25, 8.5, 6, 0, 25])
    remove_communication(d, data, x_po, y_po, x_ao, y_ao)


def remove_indonesia_communication(d, data):
    """
    Function that call remove_communication_two_regions for a specific case
    with predefined region boundaries
    """
    # Maritime continent
    x_io = np.array(
        [99, 99, 98.8, 102, 103.5, 103.5, 107.5, 114, 120, 127, 138, 138, 99]
    )
    y_io = np.array([25, 11.5, 8.8, 4.3, 2, -3, -7, -8.2, -8.6, -8.45, -8.3, 25, 25])
    x_po = np.array(
        [99, 99, 98.8, 102, 103.5, 103.5, 107.5, 114, 120, 127, 138, 138, 95, 95, 99]
    )
    y_po = np.array(
        [25, 11.5, 8.8, 4.3, 2, -3, -7, -8.2, -8.6, -8.45, -8.3, -22, -22, 25, 25]
    )
    remove_communication(d, data, x_io, y_io, x_po, y_po)


def restrict_to_subregion(data, tm, region):
    """Extract a subregion from the global transition matrix

    Args:
        data: dataset object (contains x0, xt, y0, yt)
        tm: transition matrix object
        region: ['Atlantic Ocean', 'Atlantic Ocean extended', 'Pacific Ocean', 'Indian Ocean']
    """
    if region == "Atlantic Ocean":
        xr = np.array(
            [
                -104,
                -104,
                -98,
                -84.99,
                -83.19,
                -81.06,
                -79.36,
                -77.55,
                -60,
                -72,
                -72,
                22,
                22,
                -104,
            ]
        )
        yr = np.array(
            [90, 25.5, 18, 13.4, 9.16, 8.23, 9.35, 8.31, -10.5, -54, -90, -90, 90, 90]
        )
        xyr = [[xr, yr]]
    elif region == "Atlantic Ocean extended":
        xr = np.array(
            [
                -104,
                -104,
                -98,
                -84.99,
                -83.19,
                -81.06,
                -79.36,
                -77.55,
                -60,
                -72,
                -72,
                51,
                51,
                -104,
            ]
        )
        yr = np.array(
            [90, 25.5, 18, 13.4, 9.16, 8.23, 9.35, 8.31, -10.5, -54, -90, -90, 90, 90]
        )
        xyr = [[xr, yr]]
    elif region == "Pacific Ocean":
        xr = np.array(
            [
                -104,
                -104,
                -98,
                -84.99,
                -83.19,
                -81.06,
                -79.36,
                -77.55,
                -60,
                -72,
                -72,
                -180,
                -180,
                -104,
            ]
        )
        yr = np.array(
            [90, 25.5, 18, 13.4, 9.16, 8.23, 9.35, 8.31, -10.5, -54, -90, -90, 90, 90]
        )
        xr2 = np.array([98, 98, 102.4, 108.8, 125.7, 125.7, 135, 135, 180.1, 180.1, 98])
        yr2 = np.array([90, 26, -0.7, -7.21, -8.7, -23, -23, -90, -90, 90, 90])
        xyr = [[xr, yr], [xr2, yr2]]
    elif region == "Indian Ocean":
        xr = np.array([98, 102.4, 108.8, 125.7, 125.7, 135, 135, 22, 22, 26, 98])
        yr = np.array([26, -0.7, -7.21, -8.7, -23, -23, -90, -90, -20, 31, 26])
        xyr = [[xr, yr]]
    else:
        print(
            "Available regions are: ['Atlantic Ocean', 'Atlantic Ocean extended', 'Pacific Ocean', 'Indian Ocean']"
        )
        return

    d = tm.domain
    # search bins inside region
    ids = np.empty(0, dtype="int")
    s = np.zeros_like(data.x0, dtype="bool")
    for xy in xyr:
        ids = np.append(ids, bins_in_contour(d, xy[0], xy[1]))
        s = np.logical_or(s, segments_in_contour(data, xy[0], xy[1]))
    ids = np.unique(ids)

    # Remove the bins with 0 data point in the region from the list of bin indices (ids)
    # which is important for Panama region where points
    # inside a bin might only be in another region
    bins = d.find_element(data.x0[s], data.y0[s])
    points_per_bin = np.bincount(bins[bins > -1])
    empty_bins = ismember(np.where(points_per_bin == 0)[0], ids)
    ids = np.delete(ids, empty_bins[empty_bins > -1])

    # fi is defined has everything coming from P⊄Region
    o_ids = np.setdiff1d(np.arange(0, len(tm.P)), ids)
    tm.fi = np.sum(tm.P[np.ix_(o_ids, ids)], 0)
    if np.sum(tm.fi) > 0:
        tm.fi = tm.fi / np.sum(tm.fi)
    else:
        tm.fi = np.zeros(len(ids))

    # restrict to the ids inside the region
    d = tm.domain
    tm.N = len(ids)
    tm.P = tm.P[np.ix_(ids, ids)]
    tm.M = tm.M[ids]
    tm.B = tm.B[ids]
    d.bins = d.bins[ids, :]
    d.id_og = d.id_og[ids]
    tm.fo = 1 - np.sum(tm.P, 1)
    tm.largest_connected_components()
    tm.left_and_right_eigenvectors()


def export_nc(filename, data, mat, nirvana_state=False, debug=False):
    """
    Output calculation to netCDF file

    Args:
        filename: netCDF output file
        data: trajectory's object
        dom: physical domain's object
        mat: transition matrix's object
        nirvana_state: if a nirvana state is present in the transition matrix
        debug: read and print the output file after writing to disk
    Returns: None
    """
    # segments data
    T = data.T
    x0 = data.x0
    y0 = data.y0
    xt = data.xt
    yt = data.yt

    # variables related to physical domain
    dom = mat.domain
    N0 = dom.N0
    bins = dom.bins
    id = dom.id
    id_og = dom.id_og
    coords = dom.coords
    nx = dom.nx
    ny = dom.ny
    dx = dom.dx
    dy = dom.dy
    vx = dom.vx
    vy = dom.vy
    lon = dom.lon
    lat = dom.lat

    # variables related to the transition matrix
    N = mat.N
    B = mat.B
    eigL = mat.eigL
    eigR = mat.eigR
    M = mat.M
    L = mat.L
    R = mat.R
    P = mat.P
    fi = mat.fi
    fo = mat.fo

    # delete if file exist
    if Path(filename).is_file():
        os.remove(filename)

    # basic description and variable dimensions
    f = Dataset(filename, "w", format="NETCDF4")
    f.history = "Created " + datetime.today().strftime("%d/%m/%y")
    f.description = "Transition matrix from pygtm"
    # f.set_always_mask(False) # don't use masked_array by default
    f.createDimension("N", N)
    f.createDimension("N+1", N + 1)
    f.createDimension("N0", N0)
    f.createDimension("nx", nx)
    f.createDimension("ny", ny)
    f.createDimension("gridnx", nx - 1)
    f.createDimension("gridny", ny - 1)
    f.createDimension("pointsPerBin", 4)
    f.createDimension("2", 2)
    f.createDimension("nbSegments", len(x0))
    f.createDimension("nbCoords", nx * ny)
    f.createDimension("nbParts", np.sum(M))

    if np.all(mat.R is None) or np.all(mat.L is None):
        nbEigenvalues = 0
    else:
        nbEigenvalues = min(np.shape(R)[1], np.shape(L)[1])
        if np.shape(R)[1] != np.shape(L)[1]:
            R = R[:, :nbEigenvalues]
            L = L[:, :nbEigenvalues]
            eigL = eigL[:nbEigenvalues]
            eigR = eigR[:nbEigenvalues]
    f.createDimension("nbEigenvalues", nbEigenvalues)

    # create variables in netCDF file
    # scalars
    dx_ = f.createVariable("dx", dx.dtype)
    dy_ = f.createVariable("dy", dy.dtype)
    T_ = f.createVariable("T", type(T))
    lonmin_ = f.createVariable("lonmin", "d")
    lonmax_ = f.createVariable("lonmax", "d")
    latmin_ = f.createVariable("latmin", "d")
    latmax_ = f.createVariable("latmax", "d")

    # vectors
    id_og_ = f.createVariable("id_og", id_og.dtype, "N")
    fi_ = f.createVariable("fi", fi.dtype, "N")
    fo_ = f.createVariable("fo", fo.dtype, "N")
    vx_ = f.createVariable("vx", vx.dtype, "nx")
    vy_ = f.createVariable("vy", vy.dtype, "ny")
    x0_ = f.createVariable("x0", x0.dtype, "nbSegments")
    y0_ = f.createVariable("y0", y0.dtype, "nbSegments")
    xt_ = f.createVariable("xt", xt.dtype, "nbSegments")
    yt_ = f.createVariable("yt", yt.dtype, "nbSegments")
    eigL_ = f.createVariable("eigL", "d", ("nbEigenvalues"))
    eigR_ = f.createVariable("eigR", "d", ("nbEigenvalues"))

    # transform 2d list into vector
    # we can use M to know what particles belong to what bins
    Bv = np.zeros(np.sum(M))
    j = 0
    for i in range(0, len(B)):
        Bv[j : j + M[i]] = B[i]
        j += M[i]
    Bv_ = f.createVariable("Bv", Bv.dtype, ("nbParts"))

    # matrices
    bins_ = f.createVariable("bins", bins.dtype, ("N", "pointsPerBin"))
    id_ = f.createVariable("id", id.dtype, ("gridny", "gridnx"))
    coords_ = f.createVariable("coords", coords.dtype, ("nbCoords", "2"))
    M_ = f.createVariable("M", M.dtype, ("N"))
    L_ = f.createVariable("L", "d", ("N", "nbEigenvalues"))
    R_ = f.createVariable("R", "d", ("N", "nbEigenvalues"))

    # one more state in P in the presence of a Nirvana state
    if nirvana_state:
        P_ = f.createVariable("P", P.dtype, ("N+1", "N+1"))
    else:
        P_ = f.createVariable("P", P.dtype, ("N", "N"))

    # set data
    id_og_[:] = id_og
    dx_[:] = dx
    dy_[:] = dy
    T_[:] = T
    lonmin_[:] = lon[0]
    lonmax_[:] = lon[1]
    latmin_[:] = lat[0]
    latmax_[:] = lat[1]

    vx_[:] = vx
    vy_[:] = vy
    x0_[:] = x0
    y0_[:] = y0
    xt_[:] = xt
    yt_[:] = yt
    Bv_[:] = Bv
    fi_[:] = fi
    fo_[:] = fo

    bins_[:] = bins
    id_[:] = id
    coords_[:] = coords
    P_[:] = P
    M_[:] = M
    L_[:] = L
    R_[:] = R
    eigL_[:] = eigL
    eigR_[:] = eigR

    # set units
    dx_.units = "degrees"
    dy_.units = "degrees"
    lonmin_.units = "degrees"
    lonmax_.units = "degrees"
    latmin_.units = "degrees"
    latmax_.units = "degrees"
    vx_.units = "meridional degrees"
    vy_.units = "zonal degrees"
    x0_.units = "meridional degrees"
    xt_.units = "meridional degrees"
    y0_.units = "zonal degrees"
    yt_.units = "zonal degrees"
    coords_.units = "coordinates degrees"
    T_.units = "days"
    f.close()

    # plot description for validation
    if debug:
        filename = os.path.expanduser(filename)
        f = Dataset(filename, "r")
        print(f.variables)
        f.close()


def import_nc(filename):
    """
    Import and recreate the pygtm objects from an outputed file (using export_nc)
    Args:
        filename: netCDF input file
    Output:
        data: trajectory's object
        dom: physical domain's object
        mat: transition matrix's object
    """

    # open file
    filename = os.path.expanduser(filename)
    f = Dataset(filename, "r")
    f.set_always_mask(False)  # don't return all variable as np.ma

    # basic domain params
    lon = [float(f["lonmin"][...]), float(f["lonmax"][...])]
    lat = [float(f["latmin"][...]), float(f["latmax"][...])]
    spatial_dis = len(f["vx"][...])

    # this is fast so we can just recreate
    # and update the element that were removed
    dom = physical.physical_space(lon, lat, spatial_dis)
    dom.bins = f["bins"][...].astype("int")
    dom.id = f["id"][...].astype("int")
    dom.id_og = f["id_og"][...].astype("int")

    # fill data object
    # segments are saved but not original trajectories
    data = dataset.trajectory(x=None, y=None, t=None, ids=None)
    data.T = f["T"][...]
    data.x0 = f["x0"][...]
    data.y0 = f["y0"][...]
    data.xt = f["xt"][...]
    data.yt = f["yt"][...]

    # fill matrix object
    mat = matrix.matrix_space(dom)
    mat.P = f["P"][...]
    mat.M = f["M"][...]
    mat.L = f["L"][...]
    mat.R = f["R"][...]
    mat.eigL = f["eigL"][...]
    mat.eigR = f["eigR"][...]
    mat.fi = f["fi"][...]
    mat.fo = f["fo"][...]

    # reconstruct B 2d list from Bv (vector) with M
    mat.B = []
    Bv = f["Bv"][...]
    j = 0
    for i in range(0, len(mat.M)):
        mat.B.append(Bv[j : j + mat.M[i]])
        j = j + mat.M[i]
    mat.B = np.asarray(mat.B)
    f.close()

    return data, dom, mat
