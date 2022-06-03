from pygtm import tools
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


class physical_space:
    def __init__(self, lon, lat, resolution):
        self.lon = lon
        self.lat = lat
        self.resolution = resolution

        # the bins are generated inside the limit of the domain and one variable controlling the spatial resolution
        # the number of bins in one direction equal to resolution and the other direction is chosen to build square bins
        self.nx, self.ny = self.uniform_grid(self.lon, self.lat, self.resolution)
        self.coords, self.bins, self.vx, self.vy, self.dx, self.dy = self.create_grid(
            lon, lat, self.nx, self.ny
        )

        # initialize the id of each of the bins from 1 to N0
        self.id = np.arange(0, (self.ny - 1) * (self.nx - 1))
        self.id = self.id.reshape((self.ny - 1, self.nx - 1), order="C")

        # during the algorithm if no particle is found inside bin i, it will be remove.
        # As a consequence, the probabilities of reaching (leaving) the element X is not
        # necessary located at line (column) X of the transition matrix P.
        # id_og: size evolve at the same time as self.bins
        #        allow from an element id of id[i,j] retrieved from a (lon,lat) -> id_og -> to get the index of P
        #        which is needed to initialize a tracer from a list of coordinates
        self.N0 = self.id.size
        self.id_og = np.arange(0, self.N0)

    @staticmethod
    def uniform_grid(lon, lat, size):
        """
        From a size parameters and the dimensions of the domain this
        function returns the number of bins in x-y direction to have
        bins as square as possible.
        Args:
            lon: list with limits of the boundaries in the zonal direction
            lat: list with limits of the boundaries in the meridional direction
            size: maximum number of points in one direction

        Returns:
            nx: number of divisions in the zonal direction
            ny: number of divisions in the meridional direction (to have square element)

        """
        fac = (lon[1] - lon[0]) / (lat[1] - lat[0])
        if fac > 1:
            nx = size
            ny = int(nx / fac)
        else:
            ny = size
            nx = int(ny * fac)
        return nx, ny

    @staticmethod
    def create_grid(lon, lat, nx, ny):
        """
        From two vectors x-y, this function creates a structured regular grid
        Args:
            lon: list with limits of the boundaries in the zonal direction
            lat: list with limits of the boundaries in the meridional direction
            nx: number of points in the zonal direction
            ny: number of points in the meridional direction

        Returns:
            coords: coordinates of all N points of the domain
            bins: square elements connectivity 4 coordinates index per line (per element)
                  stored in anti-anticlockwise order
            x: vector of coordinates in the zonal direction
            y: vector of coordinates in the meridional direction
            dx: size of the zonal grid
            dy: size of the meridional grid
        """
        x = np.linspace(lon[0], lon[1], num=nx, endpoint=True)
        y = np.linspace(lat[0], lat[1], num=ny, endpoint=True)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        coords = np.array(np.meshgrid(x, y)).reshape(2, -1).T
        bins = np.empty(((nx - 1) * (ny - 1), 4), dtype="uint16")
        for i in range(0, ny - 1):
            for j in range(0, nx - 1):
                n1 = i * nx + j
                n2 = n1 + 1
                n3 = (i + 1) * nx + j
                n4 = n3 + 1
                bins[i * (nx - 1) + j] = [n1, n2, n3, n4]
        return coords, bins, x, y, dx, dy

    def find_element(self, x, y):
        """
        Find element(s) where the point(s) defined by x-y is(are) located
        Args:
            x: longitude(s) of point to search
            y: latitude(s) of point to search

        Returns:
            el_list: element number where the point(s) (x_i, y_i) is(are) located
        """
        # left: a[i - 1] < v <= a[i]
        id_i = np.searchsorted(self.vy, y, side="left") - 1
        id_j = np.searchsorted(self.vx, x, side="left") - 1

        # modify for elements on one side of the boundaries
        # right: a[i - 1] <= v < a[i]
        if np.any((y == self.vy[0])):
            id_i[y == self.vy[0]] = (
                np.searchsorted(self.vy, y[y == self.vy[0]], side="right") - 1
            )
        if np.any((x == self.vx[0])):
            id_j[x == self.vx[0]] = (
                np.searchsorted(self.vx, x[x == self.vx[0]], side="right") - 1
            )

        # make sure id_i and id_j inside the domain
        keep = np.all(
            (id_i >= 0, id_i < self.ny - 1, id_j >= 0, id_j < self.nx - 1), axis=0
        )
        id_i, id_j = tools.filter_vector([id_i, id_j], keep)

        if np.isscalar(x):
            el_list = np.ones(1, dtype=int) * -1
        else:
            el_list = np.ones_like(x, dtype=int) * -1

        # get id from the grid
        el_list[keep] = self.id[id_i, id_j]

        # finally update the number to account for removed elements
        el_list = tools.ismember(el_list, self.id_og)
        return el_list

    def vector_to_matrix(self, vector):
        """
        The vector contains value at elements of the domain and not on land this function convert the vector
         (or list of vectors) to a matrix that can be plot with pcolormesh(), contourf() or other similar functions.
        Args:
            vector:  Numpy array or list of Numpy array
        """
        # we remove nirvana state if present
        if len(vector) == len(self.bins) + 1:
            vector = vector[:-1]

        mat = np.full((self.nx - 1) * (self.ny - 1), np.nan)
        mat[self.id_og] = vector
        return np.ma.masked_invalid(mat.reshape((self.ny - 1, self.nx - 1)))

    def bins_contour(self, ax, edgecolor="k", bin_id=None, projection=None):
        """
        Plot all element bins on one axis
        Args:
            ax: axis to plot on
            edgecolor: bins contour color
            bin_id: which bins to plot (default all)
            projection: add transform keyword to convert to cartopy projection
        """
        if bin_id is None:
            bins = self.bins
        else:
            bins = self.bins[bin_id]

        patches = []
        for b_i in bins:
            # corners and width/height of element
            c = (self.coords[b_i[0]][0], self.coords[b_i[0]][1])
            w = self.coords[b_i[1]][0] - self.coords[b_i[0]][0]
            h = self.coords[b_i[2]][1] - self.coords[b_i[0]][1]
            patches.append(
                Rectangle(c, w, h, edgecolor=edgecolor, fill=False, linewidth=0.5)
            )

        if projection is not None:
            p = PatchCollection(patches, transform=projection, match_original=True)
        else:
            p = PatchCollection(patches, match_original=True)

        ax.add_collection(p)
        return
