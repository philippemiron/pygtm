import numpy as np
from scipy.interpolate import interp1d


class split_trajectories:
    def __init__(self, T, x, y, t, ids):

        if T > 0:
            x0, y0, xT, yT = self.segments(T, x, y, t, ids)
        else:
            xT, yT, x0, y0 = self.segments(abs(T), x, y, t, ids)

        self.x0 = x0
        self.y0 = y0
        self.xT = xT
        self.yT = yT

    @staticmethod
    def monotonic(x):
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
        """Interpolation function
        x(t), y(t) describe the locations at time t of a trajectory
        x,y: known location
        s: oversampling coefficient (1: daily interpolation, 2: bidaily, 12: every 2h, etc.)
        xi(ti), yi(ti): interpolated location at time ti
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
        :param x1: longitude on one side of the ±180 meridian
        :param x2: longitude on the other side of the ±180 meridian
        :return: the ratio between x1 and ±180 meridian over x1 and x2
        """
        if x1 < 0:
            return (x1 + 180) / (360 - np.abs(x1 - x2))
        else:
            return (180 - x1) / (360 - np.abs(x1 - x2))

    def segments(self, T, x, y, t, index):
        """
        :param T: transition time which is the length of each segment
        :param x: list of longitude
        :param y: list of latitude
        :param t: list of time
        :param index: list same length of x,y,t to identify drifter change
        :return: x0, y0, xT, yT for each drifters where for all i: (x0[i], y0[i]) and (xT[i], yT[i]) are separated by T
        """

        oversampling = 1  # times per days
        offset = oversampling * T

        # real size loop all trajectories, count total days, multiply by oversampling
        # defined a big vector but output only 0:ptr at the end of the functions
        # should but fine but in theory can still be too short and crash on low frequency trajectories
        x0 = np.zeros(len(x) * oversampling * 20)
        y0 = np.zeros(len(x) * oversampling * 20)
        xT = np.zeros(len(x) * oversampling * 20)
        yT = np.zeros(len(x) * oversampling * 20)

        # create index where we change drifter in x,y,t
        I = np.where(abs(np.diff(index, axis=0)) > 0)[0]
        I = np.insert(I, [0, len(I)], [-1, len(index) - 1])

        # loop each trajectory
        ptr = 0
        for j in range(0, len(I) - 1):
            range_j = range(I[j] + 1, I[j + 1] + 1)
            t_j = t[range_j]
            days = np.floor(t_j[-1] - t_j[0])
            if days >= T:
                xd = x[range_j]
                yd = y[range_j]

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
                    t_i, x_i, y_i = self.trajectory_interpolation(t_j, xd, yd, oversampling)
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
                            r = self.intersection_ratio(xs[-1], x[next_p])
                            # add values at 180°
                            xs = np.insert(xs, len(xs), np.sign(xs[-1]) * 180)
                            ys = np.insert(ys, len(ys), ys[-1] + r * diff_y[next_p - 1])
                            ts = np.insert(ts, len(ts), ts[-1] + r * diff_t[next_p - 1])

                        elif i == len(idc) - 2:  # last segment cross beginning
                            r = self.intersection_ratio(x[prev_p], xs[0])
                            # add values at 180°
                            xs = np.insert(xs, 0, np.sign(xs[0]) * 180)
                            ys = np.insert(ys, 0, ys[0] - (1 - r) * diff_y[prev_p])
                            ts = np.insert(ts, 0, ts[0] - (1 - r) * diff_t[prev_p])

                        else:  # middle segments crosses back and forth
                            r1 = self.intersection_ratio(x[prev_p], xs[0])
                            r2 = self.intersection_ratio(xs[-1], x[next_p])
                            # add values at 180°
                            xs = np.insert(xs, [0, len(xs)], [np.sign(xs[0]) * 180, np.sign(xs[-1]) * 180])
                            ys = np.insert(ys, [0, len(ys)],
                                           [ys[0] - (1 - r1) * diff_y[prev_p], ys[-1] + r2 * diff_y[next_p - 1]])
                            ts = np.insert(ts, [0, len(ts)],
                                           [ts[0] - (1 - r1) * diff_t[prev_p], ts[-1] + r2 * diff_t[next_p - 1]])

                        # interpolate and add the list for this trajectory
                        tsi, xsi, ysi = self.trajectory_interpolation(ts, xs, ys, oversampling)
                        x_i = np.append(x_i, xsi)
                        y_i = np.append(y_i, ysi)
                        t_i = np.append(t_i, tsi)

                # add segments points to global list
                length = len(x_i) - offset
                if x_i.size:
                    x0[ptr:ptr + length] = x_i[0:-offset]
                    y0[ptr:ptr + length] = y_i[0:-offset]
                    xT[ptr:ptr + length] = x_i[offset:]
                    yT[ptr:ptr + length] = y_i[offset:]
                    ptr += length

        x0 = x0[:ptr]
        y0 = y0[:ptr]
        xT = xT[:ptr]
        yT = yT[:ptr]
        return x0, y0, xT, yT
