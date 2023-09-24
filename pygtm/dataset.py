import numpy as np
from scipy.interpolate import interp1d


class trajectory:
    def __init__(self, x, y, t, ids):
        # store trajectory data
        self.x = x
        self.y = y
        self.t = t
        self.ids = ids
        self.T = None
        self.x0 = None
        self.y0 = None
        self.xt = None
        self.yt = None

    @staticmethod
    def monotonic(x):
        """
        Test if an array is in a monotonic order
        Args:
            x: array

        Returns:
            True or False
        """
        # True if array is monotonic.
        # monotonic(X) returns True (False) if X is (not) monotonic.
        if len(x) == 1:
            return False

        if x[0] - x[1] > 0:
            x = np.flip(x, 0)

        dif = np.diff(x)
        if np.any(dif - np.abs(dif)):
            return False
        else:
            return True

    @staticmethod
    def trajectory_interpolation(t, x, y, s):
        """
        Interpolation function x(t), y(t) describe the locations at time t of a trajectory
        Args:
            t: time t of a trajectory
            x: longitude of the trajectory
            y: latitude of the trajectory
            s: oversampling coefficient (1: daily interpolation, 2: bidaily, 12: every 2h, etc.)

        Returns:
            ti: time of the interpolated trajectory
            fx(ti): longitude of the trajectory at the interpolated time
            fy(ti): latitude of the trajectory at the interpolated time
        """
        # interpolation functions
        fx = interp1d(t, x)
        fy = interp1d(t, y)

        # time where values will be interpolated
        days = np.floor(t[-1] - t[0])
        ti = np.linspace(t[0], t[0] + days, int(s * days + 1))
        return ti, fx(ti), fy(ti)

    @staticmethod
    def intersection_ratio(x1, x2):
        """
        Function used to interpolate trajectories at the ±180 meridian
        Args:
            x1: longitude on one side of the ±180 meridian
            x2: longitude on the other side of the ±180 meridian

        Returns:
            the ratio between x1 and ±180 meridian over x1 and x2
        """
        if x1 < 0:
            return (x1 + 180) / (360 - np.abs(x1 - x2))
        else:
            return (180 - x1) / (360 - np.abs(x1 - x2))

    def create_segments(self, T):
        """
        Subdivide full trajectories into a list of segments of T days
        Args:
            T: transition time

        Returns:
            x0: longitude of each segments at time t0
            y0: latitude of each segments at time t0
            xt: longitude of each segments at time t0+T
            yt: latitude of each segments at time t0+T

        """
        self.T = T
        oversampling = 1  # times per days
        offset = oversampling * abs(T)

        # real size loop all trajectories, count total days, multiply by oversampling
        # defined a big vector but output only 0:ptr at the end of the functions
        # should but fine but in theory can still be too short and crash on low frequency trajectories
        x0 = np.zeros(len(self.x) * oversampling * 20)
        y0 = np.zeros(len(self.x) * oversampling * 20)
        xt = np.zeros(len(self.x) * oversampling * 20)
        yt = np.zeros(len(self.x) * oversampling * 20)

        # create index where we change drifter in x,y,t
        I = np.where(abs(np.diff(self.ids, axis=0)) > 0)[0]
        I = np.insert(I, [0, len(I)], [-1, len(self.ids) - 1])

        # loop each trajectory
        ptr = 0
        for j in range(0, len(I) - 1):
            range_j = range(I[j] + 1, I[j + 1] + 1)
            t_j = self.t[range_j]
            days = np.floor(t_j[-1] - t_j[0])
            if days >= abs(T):
                xd = self.x[range_j]
                yd = self.y[range_j]

                # make sure it is ordered correctly
                if not self.monotonic(t_j):
                    order = np.argsort(t_j)
                    t_j = t_j[order]
                    xd = xd[order]
                    yd = yd[order]

                # Because we are interpolating we have to be careful when drifters cross the dateline ±180 in that case
                # we split the trajectory into segments that we individually interpolate then put back together

                # look for ±180 crossing by finding jump in longitude larger than 180°
                diff_x = np.diff(xd)
                idc = np.where(np.abs(diff_x) > 180)[0]

                if len(idc) == 0:
                    # normally interpolate the whole trajectories
                    t_i, x_i, y_i = self.trajectory_interpolation(
                        t_j, xd, yd, oversampling
                    )
                else:
                    # the drifter is crossing ±180 meridian
                    t_i = np.empty(0)
                    x_i = np.empty(0)
                    y_i = np.empty(0)
                    diff_t = np.diff(t_j)
                    diff_y = np.diff(yd)

                    # loop trajectories segments cutting at 180°
                    idc = np.insert(idc, [0, len(idc)], [-1, len(xd) - 1])
                    for i in range(0, len(idc) - 1):
                        ids = np.arange(idc[i] + 1, idc[i + 1] + 1)
                        xs = xd[ids]
                        ys = yd[ids]
                        ts = t_j[ids]

                        # index of points before and after the current segment
                        next_p = ids[-1] + 1
                        prev_p = ids[0] - 1

                        # If we split a trajectory into two pieces before 180° and after 180°
                        # on the odd sections 180° is added at the end of the list
                        # on the even sections 180° is added at the beginning of the list
                        # three cases if we add 180° at the beginning, end or both side of the segment
                        if i == 0:  # first segment cross at the end
                            # find ratio between 180° and the to points that cross it
                            r = self.intersection_ratio(xs[-1], self.x[next_p])
                            # add values at 180°
                            xs = np.insert(xs, len(xs), np.sign(xs[-1]) * 180)
                            ys = np.insert(ys, len(ys), ys[-1] + r * diff_y[next_p - 1])
                            ts = np.insert(ts, len(ts), ts[-1] + r * diff_t[next_p - 1])

                        elif i == len(idc) - 2:  # last segment cross beginning
                            r = self.intersection_ratio(self.x[prev_p], xs[0])
                            # add values at 180°
                            xs = np.insert(xs, 0, np.sign(xs[0]) * 180)
                            ys = np.insert(ys, 0, ys[0] - (1 - r) * diff_y[prev_p])
                            ts = np.insert(ts, 0, ts[0] - (1 - r) * diff_t[prev_p])

                        else:  # middle segments crosses back and forth
                            r1 = self.intersection_ratio(self.x[prev_p], xs[0])
                            r2 = self.intersection_ratio(xs[-1], self.x[next_p])
                            # add values at 180°
                            xs = np.insert(
                                xs,
                                [0, len(xs)],
                                [np.sign(xs[0]) * 180, np.sign(xs[-1]) * 180],
                            )
                            ys = np.insert(
                                ys,
                                [0, len(ys)],
                                [
                                    ys[0] - (1 - r1) * diff_y[prev_p],
                                    ys[-1] + r2 * diff_y[next_p - 1],
                                ],
                            )
                            ts = np.insert(
                                ts,
                                [0, len(ts)],
                                [
                                    ts[0] - (1 - r1) * diff_t[prev_p],
                                    ts[-1] + r2 * diff_t[next_p - 1],
                                ],
                            )

                        # interpolate and add the list for this trajectory
                        tsi, xsi, ysi = self.trajectory_interpolation(
                            ts, xs, ys, oversampling
                        )
                        x_i = np.append(x_i, xsi)
                        y_i = np.append(y_i, ysi)
                        t_i = np.append(t_i, tsi)

                # add segments points to global list
                length = len(x_i) - offset
                if x_i.size:
                    x0[ptr : ptr + length] = x_i[0:-offset]
                    y0[ptr : ptr + length] = y_i[0:-offset]
                    xt[ptr : ptr + length] = x_i[offset:]
                    yt[ptr : ptr + length] = y_i[offset:]
                    ptr += length

        x0 = x0[:ptr]
        y0 = y0[:ptr]
        xt = xt[:ptr]
        yt = yt[:ptr]

        if T > 0:
            self.x0 = x0
            self.y0 = y0
            self.xt = xt
            self.yt = yt
        else:
            # invert the points
            self.x0 = xt
            self.y0 = yt
            self.xt = x0
            self.yt = y0

    def filtering(self, x_range=None, y_range=None, t_range=None, complete_track=True):
        """
        Returns trajectory in a spatial domain and/or temporal range
        Args:
            x_range: longitudinal range (ascending order)
            y_range: meridional range (ascending order)
            t_range: temporal range (ascending order)
            complete_track: True: full trajectories is plotted
                            False: trajectories is plotted after it reaches the region

        Returns:
            segs [Ns, 2, 2]: list of segments to construct a LineCollection object
                - i: segment number (Ns)
                - j: coordinates of the beginning (0) or end (1) of the segment i
                - k: longitude (0) or latitude (1) of the coordiates j
            segs_t [Ns]: time associated to each segments
            segs_ind [Nt,2]: indices of the first and last segment of a trajectory
                     ex: trajectory 0 contains the segs[segs_ind[0,0]:segs_ind[0,1]]
        """
        if x_range is None:
            x_range = [np.min(self.x), np.max(self.x)]
        if y_range is None:
            y_range = [np.min(self.y), np.max(self.y)]
        if t_range is None:
            t_range = [np.min(self.t), np.max(self.t)]

        # identified drifters change
        I = np.where(abs(np.diff(self.ids, axis=0)) > 0)[0]
        I = np.insert(I, [0, len(I)], [-1, len(self.ids) - 1])

        segs = np.empty((0, 2, 2))
        segs_t = np.empty(0)
        segs_ind = np.zeros((0, 2), dtype="int")
        for j in range(0, len(I) - 1):
            range_j = np.arange(I[j] + 1, I[j + 1] + 1)
            xd = self.x[range_j]
            yd = self.y[range_j]
            td = self.t[range_j]

            # keep trajectory inside the specific domain and time frame
            keep = np.logical_and.reduce(
                (
                    xd >= x_range[0],
                    xd <= x_range[1],
                    yd >= y_range[0],
                    yd <= y_range[1],
                    td >= t_range[0],
                    td <= t_range[1],
                )
            )

            if np.sum(keep) > 1:
                # if complete_track: full trajectories is plotted
                # else: only after reaching the region
                if not complete_track:
                    reach = np.argmax(keep)
                    range_j = range_j[reach:]
                    xd = xd[reach:]
                    yd = yd[reach:]
                    td = td[reach:]

                if len(xd) > 1:
                    # search for trajectories crossing the ±180
                    cross_world = np.where(np.diff(np.sign(xd)))[0]
                    cross_world = np.unique(
                        np.insert(cross_world, [0, len(cross_world)], [-1, len(xd) - 1])
                    )

                    for k in range(0, len(cross_world) - 1):
                        ind = np.arange(cross_world[k] + 1, cross_world[k + 1] + 1)

                        if len(ind) > 1:
                            # reorganize array for LineCollection
                            pts = np.array([xd[ind], yd[ind]]).T.reshape(-1, 1, 2)
                            segs_i = np.concatenate([pts[:-1], pts[1:]], axis=1)

                            if len(segs_i) > 1:
                                segs_t_i = np.convolve(
                                    td, np.repeat(1.0, 2) / 2, "valid"
                                )  # average per segment
                            else:
                                segs_t_i = td

                            segs_ind = np.vstack(
                                (
                                    segs_ind,
                                    np.array([len(segs), len(segs) + len(segs_i)]),
                                )
                            )
                            segs = np.concatenate((segs, segs_i), axis=0)
                            segs_t = np.concatenate((segs_t, segs_t_i), axis=0)
        return segs, segs_t, segs_ind
