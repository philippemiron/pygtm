from pygtm import tools
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes

class physical_space:
    def __init__(self, lon, lat, resolution):
        self.lon = lon
        self.lat = lat
        self.resolution = resolution

        # the bins are generated inside the limit of the domain and one variable controlling the spatial resolution
        # the number of bins in one direction equal to resolution and the other direction is chosen to build square bins
        self.nx, self.ny = self.uniform_grid(self.lon, self.lat, self.resolution)

        self.coords, self.bins, self.vx, self.vy, self.dx, self.dy = self.create_grid(lon, lat, self.nx, self.ny)

        # initialize the id of each of the bins from 1 to N0
        self.id = np.arange(0, (self.ny - 1) * (self.nx - 1))
        self.id = self.id.reshape((self.ny - 1, self.nx - 1), order='C')

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
        """From a size parameters and the dimensions of the domain this
        function returns the number of bins in x-y direction to have
        bins as square as possible
        lon, lat: list with limits of the boundaries in the zonal direction and the meridional direction
        size: maximum number of divisions in one direction
        nx, ny: number of divisions in both direction
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
        """From two vectors x-y, this function creates a structured regular grid
        lon, lat: list with limits of the boundaries in the zonal direction and the meridional direction
        nx, ny: number of points in the zonal direction and the meridional direction
        Returns
        coords (number_nodes): list of all points from meshgrid output
        elements (number_elements, 4): square elements connectivity
                 4 coordinates index per line (per element) stored in anti-anticlockwise order
        x, y: vector of coordinates in the zonal direction and the meridional direction
        dx, dy: size of the grid
        """
        x = np.linspace(lon[0], lon[1], num=nx, endpoint=True)
        y = np.linspace(lat[0], lat[1], num=ny, endpoint=True)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        coords = np.array(np.meshgrid(x, y)).reshape(2, -1).T
        elements = np.empty(((nx - 1) * (ny - 1), 4), dtype='uint16')
        for i in range(0, ny - 1):
            for j in range(0, nx - 1):
                n1 = i * nx + j
                n2 = n1 + 1
                n3 = (i + 1) * nx + j
                n4 = n3 + 1
                elements[i * (nx - 1) + j] = [n1, n2, n3, n4]
        return coords, elements, x, y, dx, dy

    def find_element(self, x, y):
        """Find element(s) where the point(s) defined by x-y is(are) located
        x-y: coordinate of one or multiple points to search"""

        # left: a[i - 1] < v <= a[i]
        id_i = np.searchsorted(self.vy, y, side='left') - 1
        id_j = np.searchsorted(self.vx, x, side='left') - 1

        # modify for elements on one side of the boundaries
        # right: a[i - 1] <= v < a[i]
        if np.any((y == self.vy[0])):
            id_i[y == self.vy[0]] = np.searchsorted(self.vy, y[y == self.vy[0]], side='right') - 1
        if np.any((x == self.vx[0])):
            id_j[x == self.vx[0]] = np.searchsorted(self.vx, x[x == self.vx[0]], side='right') - 1

        # make sure id_i and id_j inside the domain
        keep = np.all((id_i >= 0, id_i < self.ny - 1, id_j >= 0, id_j < self.nx - 1), axis=0)
        id_i, id_j = tools.filter_vector([id_i, id_j], keep)

        el_list = np.ones_like(x, dtype=int) * -1
        el_list[keep] = self.id[tuple(zip((id_i, id_j)))][0]

        # finally update the number to account for removed elements
        # have to look at the P construction if I include this here
        el_list = tools.ismember(el_list, self.id_og)

        return el_list

    def vector_to_matrix(self, vector):
        """The vector contains value at elements of the domain and not on land this function convert the vector
         (or list of vectors) to a matrix that can be plot with pcolormesh(), contourf() or other similar functions.
        vectors: Numpy array or list of Numpy array
        nx : number of longitude nodes in the grid
        ny : number of latitude nodes in the grid
        id_og: id_og[i] is the index of ocean element i in the full grid that includes both land and ocean
        mat: Numpy masked matrix or list of Numpy masked matrix
        """
        mat = np.full((self.nx - 1) * (self.ny - 1), np.nan)
        mat[self.id_og] = vector
        return np.ma.masked_invalid(mat.reshape((self.ny - 1, self.nx - 1)))

    def bins_contours(self, ax):
        for b_i in self.bins:
            # corners and width/height of element
            c = (self.coords[b_i[0]][0], self.coords[b_i[0]][1])
            w = self.coords[b_i[1]][0] - self.coords[b_i[0]][0]
            h = self.coords[b_i[2]][1] - self.coords[b_i[0]][1]

            if isinstance(ax, GeoAxes):
                ax.plot([c[0], c[0] + w, c[0] + w, c[0], c[0]],
                        [c[1], c[1], c[1] + h, c[1] + h, c[1]],
                        'k', linewidth=0.2, zorder=1, transform=ccrs.PlateCarree())
            else:
                ax.plot([c[0], c[0] + w, c[0] + w, c[0], c[0]],
                        [c[1], c[1], c[1] + h, c[1] + h, c[1]],
                        'k', linewidth=0.2, zorder=1)
