#!/usr/bin/env python3

import cv2
import math
import itertools
import numpy as np


def show_debug(img, grid, lines, columns, edges, warped, warped_edges,
               cdst, img_grid, img_grid_flat):
    from matplotlib import pyplot as plt
    plot_number = 0

    plot_number += 1
    plt.subplot(3, 4, plot_number)
    plt.imshow(img, cmap = 'gray')
    plt.title('Original Image')

    plot_number += 1
    plt.subplot(3, 4, plot_number)
    plt.imshow(edges, cmap = 'gray')
    plt.title('Edge Image')

    plot_number += 1
    plt.subplot(3, 4, plot_number)
    plt.imshow(cdst, cmap = 'gray')
    plt.title('Interesting lines')

    plot_number += 1
    plt.subplot(3, 4, plot_number)
    plt.imshow(warped, cmap = 'gray')
    plt.title('Warped')

    plot_number += 1
    plt.subplot(3, 4, plot_number)
    plt.imshow(warped_edges, cmap = 'gray')
    plt.title('Warped edges')

    plot_number += 1
    plt.subplot(3, 4, plot_number)
    plt.imshow(lines, cmap = 'gray')
    plt.title('Lines')

    plot_number += 1
    plt.subplot(3, 4, plot_number)
    plt.imshow(columns, cmap = 'gray')
    plt.title('Columns')

    plot_number += 1
    plt.subplot(3, 4, plot_number)
    plt.imshow(img_grid, cmap = 'gray')
    plt.title('Detected {} lines, {} rows of {}px x {}px'.
              format(len(grid.all_y), len(grid.all_x),
                     grid.xstep, grid.ystep)),

    plot_number += 1
    plt.subplot(3, 4, plot_number)
    plt.imshow(img_grid_flat, cmap = 'gray')
    plt.title('Reconstitution')

    plt.show()


def mean(x):
    return sum(x) / len(x)


class Grid(object):
    def __init__(self, edges, debug=False):
        """
        Given the result of a cv2.Canny, find
        a grid in the given image.

        The grid have no start and no end, only a "cell width" and an
        anchor (an intersection, so you can anlign the grid with the image).
        Exposes a list of columns (x, width) and a list of rows (y, height),
        as self.all_x and self.all_y.

        Exposes a all_cells() method, yielding every cells as tuples
        of (x, y, width, height).

        And a draw(self, image, color=(255, 0, 0), thickness=2) method,
        to draw the grid on a given image, usefull to check for correctness.

        """
        self.lines = lines = self.keep_lines(edges)
        self.columns = columns = self.keep_cols(edges)

        min_x_step = int(edges.shape[0] / 50)
        min_y_step = int(edges.shape[1] / 50)
        self.x, self.xstep = self.find_step(np.sum(lines, axis=1), min_x_step,
                                            debug=debug, debug_title="Lines")
        self.y, self.ystep = self.find_step(np.sum(columns, axis=0), min_y_step,
                                            debug=debug, debug_title="Columns")
        self.x = self.x % self.xstep  # Push back first point to the left
        self.y = self.y % self.ystep  # idem top the top
        self.shape = edges.shape
        self.all_x = [(x, self.xstep)
                      for x in range(self.x, self.shape[0], self.xstep)]
        self.all_y = [(y, self.ystep)
                      for y in range(self.y, self.shape[1], self.ystep)]
        if self.x < self.xstep and self.x > self.xstep / 5:
            # There is a partial column at the left:
            self.all_x.insert(0, (0, self.x))
        if self.y < self.ystep and self.y > self.ystep / 5:
            # There is a partial row at the top:
            self.all_y.insert(0, (0, self.y))

    @staticmethod
    def keep_lines(array):
        """
        Apply a sliding window to each lines, only keep pixels surrounded
        by 4 pixels, so only keep sequences of 5 pixels minimum.
        """
        out = array.copy()
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                if y > 1 and y + 2 < array.shape[1]:
                    out[x, y] = min(array[x][y - 2],
                                    array[x][y - 1],
                                    array[x][y],
                                    array[x][y + 1],
                                    array[x][y + 2])
        return out


    @staticmethod
    def keep_cols(array):
        """
        Apply a sliding window to each column, only keep pixels surrounded
        by 4 pixels, so only keep sequences of 5 pixels minimum.
        """
        out = array.copy()
        for y in range(array.shape[1]):
            for x in range(array.shape[0]):
                if x > 1 and x + 2 < array.shape[0]:
                    out[x, y] = min(array[x - 2][y],
                                    array[x - 1][y],
                                    array[x][y],
                                    array[x + 1][y],
                                    array[x + 2][y])
        return out


    @staticmethod
    def find_step(positions, min_step=4, debug_title="Lines", debug=False):
        """
        For positions, a numpy array of values, returns a
        (start, step) tuple, of the most evident repetitive pattern
        in the input.

        For a positions array like:
        [0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 100, ...
        returns (5, 5)
                  \  \_ Step
                   \___ Start

        This is done by brute force, trying each possibilities, and
        summing the values for each possibility. This returns only
        the best match.
        """
        positions = positions - mean(positions)
        positions = np.convolve(positions, (1 / 3, 2 / 3, 1 / 3))
        # if debug:
        #     with Plotting() as plt:
        #         plt.plot(positions)
        #         plt.title(debug_title)
        best = (0, 0, 0)
        for start, step, value in [(start, step, sum(positions[start::step]))
                                   for step in range(min_step, int(len(positions) / 2))
                                   for start in range(int(len(positions) / 2))]:
            if value > best[2]:
                best = start, step, value
        return best[0], best[1]


    def cells_line_by_line(self):
        """
        Return all cells, line by line, like:
        [[(x, y), (x, y), ...]
         [(x, y), (x, y), ...]
         ... ]
        """
        for x, width in self.all_x:
            yield [(x, y) for y, height in self.all_y]

    def all_cells(self):
        """
        returns tuples of (x, y, width, height) for each cell.
        """
        for x, width in self.all_x:
            for y, height in self.all_y:
                yield (x, y, width, height)

    def draw(self, image, color=(255, 0, 0), thickness=2):
        for x, width in self.all_x:
            for y, height in self.all_y:
                cv2.rectangle(image, (y, x), (y + height, x + width),
                              color, thickness)

    def __str__(self):
        return '<Grid of {} lines, {} cols, cells: {}px × {}px>'.format(
            self.y, self.x, self.xstep, self.xstep)


def splitfn(fn):
    import os
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


def experiment_using_chessboarddetect(args):
    img = cv2.imread(args.file)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#cv2.findChessboardCorners
    pattern_size = (3, 3)
    found, corners = cv2.findChessboardCorners(img, pattern_size, 1)  # cv2.CV_CALIB_CB_ADAPTIVE_THRESH)
    if found:
        term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.drawChessboardCorners(vis, pattern_size, corners, found)
    # path, name, ext = splitfn(args.file)
    # cv2.imwrite('%s/%s_chess.bmp' % ('.', name), vis)
    if not found:
        print('chessboard not found')
        return
    img_points = []
    obj_points = []
    pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    square_size = 1.0
    pattern_points *= square_size

    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)
    h, w = img.shape[:2]
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    print("RMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    return


def angle_distance(a, b):
    a = abs(math.atan2(a[3] - a[1], a[2] - a[0]) - math.atan2(b[3] - b[1], b[2] - b[0])) % (math.pi * 2)
    return math.pi * 2 - a if a > math.pi else a


def find_farest_lines(lines):
    """
    We should sort lines in two buckest: horizontals and verticals first
    then get the extrems of both buckets. We use this by computing the angle of
    each segment, then cluster them in two buckets using the following
    algorithm:

    Find two opposites values, then put every remaining value to the nearest
    bucket. But we're in a circular notation (radians) there is no notion of
    "opposite values".

    Another algorithm should be to compute the average distance (in radians) between
    each pairs, use this
    """
    a, b, c = lines.shape
    all_lines = [(lines[i][0][0], lines[i][0][1], lines[i][0][2], lines[i][0][3]) for i in range(lines.shape[0])]
    farest = sorted([(id1, id2, angle_distance(line1, line2)) for
                     ((id1, line1), (id2, line2)) in
                     itertools.combinations(enumerate(all_lines), 2)],
                    key=lambda x: x[2])[-1]
    bucket_a_center, bucket_b_center = all_lines[farest[0]], all_lines[farest[1]]
    bucket_a = []
    bucket_b = []
    for line in all_lines:
        if angle_distance(line, bucket_a_center) < angle_distance(line, bucket_b_center):
            bucket_a.append(line)
        else:
            bucket_b.append(line)
    # Possible enhancment: get the two longest from each buckets
    return bucket_a[0], bucket_a[1], bucket_b[0], bucket_b[1]


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def sort_points(a, b, c, d):
    """Sort 4 points returning them in this order:
    top-left, top-right, bottom-right, bottom-left.
    Each poins is given as an (x, y) tuple.
    """
    center = (sum(x[0] for x in (a, b, c, d)) / 4,
              sum(x[1] for x in (a, b, c, d)) / 4)
    top = []
    bottom = []
    for point in a, b, c, d:
        if point[1] < center[1]:
            top.append(point)
        else:
            bottom.append(point)
    top_left = top[1] if top[0][0] > top[1][0] else top[0]
    top_right = top[0] if top[0][0] > top[1][0] else top[1]
    bottom_left = bottom[1] if bottom[0][0] > bottom[1][0] else bottom[0]
    bottom_right = bottom[0] if bottom[0][0] > bottom[1][0] else bottom[1]
    return top_left, top_right, bottom_right, bottom_left


def main():
    from argparse import ArgumentParser
    import sys
    parser = ArgumentParser(description='Grid finder')
    parser.add_argument('file', help='Input image')
    parser.add_argument('--debug', help='Using matplotlib, display information '
                        'about each step of the process.',
                        action='store_true')
    parser.add_argument('--verbose', '-v',
                        help='Use more verbose, a bit less parsable output',
                        action='store_true')
    parser.add_argument('--json', help='Print the grid in json',
                        action='store_true')
    parser.add_argument('--term', help='Print the grid as colored brackets',
                        action='store_true')
    parser.add_argument('--imwrite', help='Write a clean image of the grid')
    args = parser.parse_args()
    # experiment_using_chessboarddetect(args)
    img = cv2.imread(args.file)
    edges = cv2.Canny(img, 100, 200)  # try 66, 133, 3 ?
    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 40, np.array([]),
                            minLineLength=200, maxLineGap=10)
    best_lines = find_farest_lines(lines)
    points = [line_intersection(((x1, x2), (y1, y2)), ((X1, X2), (Y1, Y2))) for
              (x1, x2, y1, y2), (X1, X2, Y1, Y2) in itertools.combinations(best_lines, 2)]
    points = [point for point in points if point[0] > 0 and point[1] > 0]
    for x1, y1, x2, y2 in best_lines:
        cv2.line(cdst, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
    for point in points:
        cv2.circle(cdst, (int(point[0]), int(point[1])), 10, (255, 0, 0), 3)
    quad_pts = np.array([(img.shape[1] / 4, img.shape[0] / 4),
                         (img.shape[1] * 3 / 4, img.shape[0] / 4),
                         (img.shape[1] * 3 / 4, img.shape[0] * 3 / 4),
                         (img.shape[1] / 4, img.shape[0] * 3 / 4)], np.float32)
    points = sort_points(*points)
    trans = cv2.getPerspectiveTransform(np.array(points, np.float32), quad_pts)
    warped = cv2.warpPerspective(img, trans, (img.shape[1], img.shape[0]))
    warped_edges = cv2.Canny(warped, 100, 200)  # try 66, 133, 3 ?
    grid = Grid(warped_edges, debug=args.debug)

    if args.debug:
        img_grid = warped.copy()
        grid.draw(img_grid, (255, 255, 255))
        img_grid_flat = img_grid.copy()
        for x, y, width, height in grid.all_cells():
            mean_color = cv2.mean(warped[x:x + width, y:y + height])[:3]
            img_grid_flat[x:x + width, y:y + height] = mean_color
        show_debug(img, grid, grid.lines, grid.columns, edges, warped, warped_edges, cdst, img_grid, img_grid_flat)
    if args.term:
        def print_color(*args, **kwargs):
            """
            Like print() but with extra `color` argument,
            taking (red, green, blue) tuple. (0-255).
            """
            color = kwargs['color']
            reduction = 255 / 5
            del kwargs['color']
            print('\x1b[38;5;%dm' % (16 + (int(color[0] / reduction) * 36) +
                                     (int(color[1] / reduction) * 6) +
                                     int(color[2] / reduction)), end='')
            print(*args, **kwargs)
            print('\x1b[0m', end='')

        for line in grid.cells_line_by_line():
            for cell in line:
                x, y = cell[0], cell[1]
                color = cv2.mean(img[x:x + grid.xstep, y:y + grid.ystep])
                print_color('[]', color=(color[:3]), end='')
            print()
    if args.json:
        import json
        lines = []
        for line in grid.cells_line_by_line():
            line = [(x, y, cv2.mean(img[x:x + grid.xstep,
                                        y:y + grid.ystep]))
                    for x, y in line]
            line = [{'x': cell[0],
                     'y': cell[1],
                     'color': (int(cell[2][0]),
                               int(cell[2][1]),
                               int(cell[2][2]))}
                    for cell in line]
            lines.append(line)
        print(json.dumps(lines, indent=4))
        sys.exit(0)
    if args.imwrite:
        img_flat = img.copy()
        for x, y, width, height in grid.all_cells():
            mean_color = cv2.mean(img[x:x + width, y:y + height])[:3]
            img_flat[x:x + width, y:y + height] = mean_color
        cv2.imwrite(args.imwrite, img_flat)

    if args.verbose:
        print('First column at:', grid.x)
        print('First row at:', grid.y)
        print('Column width:', grid.xstep)
        print('Row width:', grid.ystep)
    sys.exit(0)
    print(grid.x, grid.y, grid.xstep, grid.ystep)

if __name__ == '__main__':
    main()
