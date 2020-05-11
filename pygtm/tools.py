import os
import sys
from datetime import datetime
import numpy as np
from pathlib import Path
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

def export_nc(filename, data, dom, mat, nirvana_state=False, debug=False):
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
    f = Dataset(filename, 'w', format='NETCDF4')
    f.history = "Created " + datetime.today().strftime("%d/%m/%y")
    f.description = "Transition matrix from pygtm"
    #f.set_always_mask(False) # don't use masked_array by default
    f.createDimension('N', N)
    f.createDimension('N+1', N+1)
    f.createDimension('N0', N0)
    f.createDimension('nx', nx)
    f.createDimension('ny', ny)
    f.createDimension('gridnx', nx - 1)
    f.createDimension('gridny', ny - 1)
    f.createDimension('pointsPerBin', 4)
    f.createDimension('2', 2)
    f.createDimension('nbSegments', len(x0))
    f.createDimension('nbCoords', nx * ny)
    f.createDimension('nbParts', np.sum(M))

    if np.all(mat.R == None) or np.all(mat.L == None):
        nbEigenvalues = 0
    else:
        nbEigenvalues = min(np.shape(R)[1], np.shape(L)[1])
        if np.shape(R)[1] != np.shape(L)[1]:
            R = R[:, :nbEigenvalues]
            L = L[:, :nbEigenvalues]
            eigL = eigL[:nbEigenvalues]
            eigR = eigR[:nbEigenvalues]
    f.createDimension('nbEigenvalues', nbEigenvalues)

    # create variables in netCDF file
    # scalars
    dx_ = f.createVariable('dx', dx.dtype)
    dy_ = f.createVariable('dy', dy.dtype)
    T_ = f.createVariable('T', type(T))
    lonmin_ = f.createVariable('lonmin', 'd')
    lonmax_ = f.createVariable('lonmax', 'd')
    latmin_ = f.createVariable('latmin', 'd')
    latmax_ = f.createVariable('latmax', 'd')

    # vectors
    id_og_ = f.createVariable('id_og', id_og.dtype, 'N')
    fi_ = f.createVariable('fi', fi.dtype, 'N')
    fo_ = f.createVariable('fo', fo.dtype, 'N')
    vx_ = f.createVariable('vx', vx.dtype, 'nx')
    vy_ = f.createVariable('vy', vy.dtype, 'ny')
    x0_ = f.createVariable('x0', x0.dtype, 'nbSegments')
    y0_ = f.createVariable('y0', y0.dtype, 'nbSegments')
    xt_ = f.createVariable('xt', xt.dtype, 'nbSegments')
    yt_ = f.createVariable('yt', yt.dtype, 'nbSegments')
    eigL_ = f.createVariable('eigL', 'd', ('nbEigenvalues'))
    eigR_ = f.createVariable('eigR', 'd', ('nbEigenvalues'))

    # transform 2d list into vector
    # we can use M to know what particles belong to what bins
    Bv = np.zeros(np.sum(M))
    j = 0
    for i in range(0, len(B)):
        Bv[j:j + M[i]] = B[i]
        j += M[i]
    Bv_ = f.createVariable('Bv', Bv.dtype, ('nbParts'))

    # matrices
    bins_ = f.createVariable('bins', bins.dtype, ('N', 'pointsPerBin'))
    id_ = f.createVariable('id', id.dtype, ('gridny', 'gridnx'))
    coords_ = f.createVariable('coords', coords.dtype, ('nbCoords', '2'))
    M_ = f.createVariable('M', M.dtype, ('N'))
    L_ = f.createVariable('L', 'd', ('N', 'nbEigenvalues'))
    R_ = f.createVariable('R', 'd', ('N', 'nbEigenvalues'))

     # one more state in P in the presence of a Nirvana state
    if nirvana_state:
        P_ = f.createVariable('P', P.dtype, ('N+1', 'N+1'))
    else:
        P_ = f.createVariable('P', P.dtype, ('N', 'N'))

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
    dx_.units = 'degrees'
    dy_.units = 'degrees'
    lonmin_.units = 'degrees'
    lonmax_.units = 'degrees'
    latmin_.units = 'degrees'
    latmax_.units = 'degrees'
    vx_.units = 'meridional degrees'
    vy_.units = 'zonal degrees'
    x0_.units = 'meridional degrees'
    xt_.units = 'meridional degrees'
    y0_.units = 'zonal degrees'
    yt_.units = 'zonal degrees'
    coords_.units = 'coordinates degrees'
    T_.units = 'days'
    f.close()
    
    # plot description for validation
    if debug:
        filename = os.path.expanduser(filename)
        f = Dataset(filename, 'r')
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
    f = Dataset(filename, 'r')
    f.set_always_mask(False) # don't return all variable as np.ma

    # basic domain params
    lon = [float(f['lonmin'][...]), float(f['lonmax'][...])]
    lat = [float(f['latmin'][...]), float(f['latmax'][...])]
    spatial_dis = len(f['vx'][...])

    # this is fast so we can just recreate
    # and update the element that were removed
    dom = physical.physical_space(lon, lat, spatial_dis)
    dom.bins = f['bins'][...].astype('int')
    dom.id = f['id'][...].astype('int')
    dom.id_og = f['id_og'][...].astype('int')

    # fill data object
    # segments are saved but not original trajectories
    data = dataset.trajectory(x=None, y=None, t=None, ids=None)
    data.T = f['T'][...]
    data.x0 = f['x0'][...]
    data.y0 = f['y0'][...]
    data.xt = f['xt'][...]
    data.yt = f['yt'][...]

    # fill matrix object
    mat = matrix.matrix_space(dom)
    mat.P = f['P'][...]
    mat.M = f['M'][...]
    mat.L = f['L'][...]
    mat.R = f['R'][...]
    mat.eigL = f['eigL'][...]
    mat.eigR = f['eigR'][...]
    mat.fi = f['fi'][...]
    mat.fo = f['fo'][...]

    # reconstruct B 2d list from Bv (vector) with M    
    mat.B = []
    Bv = f['Bv'][...]
    j = 0
    for i in range(0, len(mat.M)):
        mat.B.append(Bv[j:j + mat.M[i]])
        j = j + mat.M[i]
    mat.B = np.asarray(mat.B)
    f.close()

    return data, dom, mat
