#!/usr/bin/env python3

"""Try to find a grid (or a chessboard) in the given image.
"""

import sys
import cv2
import math
import itertools
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt

class GridFinderException(Exception):
    """Exception thrown by GridFinder when somethings wrong"""
    pass


class DebugUI(object):
    """Build a debug user interface showing each step of the grid finding
    process.
    """
    def __init__(self):
        self.plot_number = 0

    @staticmethod
    def draw_target_points(edges, points):
        """Draw four circles where the image will be mapped after
        transformation.
        """
        if points is None:
            return None
        top_left, top_right, bottom_right, _ = points
        width = norm(np.array(top_left, np.float32) -
                     np.array(top_right, np.float32))
        height = norm(np.array(top_right, np.float32) -
                      np.array(bottom_right, np.float32))
        quad_pts = np.array([top_left,
                             (top_left[0] + width, top_left[1]),
                             (top_left[0] + width, top_left[1] + height),
                             (top_left[0], top_left[1] + height)], np.float32)
        img_target_points = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for point in quad_pts:
            cv2.circle(img_target_points, (int(point[0]), int(point[1])),
                       2, (255, 0, 0), 3)
        return img_target_points

    @staticmethod
    def draw_interesting_lines(edges, lines, points=None):
        """Show the interesting lines found to find the grid.
        """
        img_interesting_lines = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in lines:
            cv2.line(img_interesting_lines, (x1, y1), (x2, y2), (0, 0, 255), 3,
                     cv2.LINE_AA)
        if points is not None:
            for point in points:
                cv2.circle(img_interesting_lines, (int(point[0]),
                                                   int(point[1])),
                           2, (255, 0, 0), 3)
        return img_interesting_lines

    @staticmethod
    def draw_grid(warped, grid):
        """Draw the grid on top of the warped image.
        """
        if warped is None:
            return None
        img_grid = warped.copy()
        grid.draw(img_grid, (255, 255, 255))
        return img_grid

    @staticmethod
    def draw_flat_grid(warped, grid):
        """Draw the grid, then compute mean color in each cells to "clean" it.
        """
        if warped is None:
            return None
        img_grid = warped.copy()
        grid.draw(img_grid, (255, 255, 255))
        for x, y, width, height in grid.all_cells():
            img_grid[x:x + width, y:y + height] = cv2.mean(
                warped[x:x + width, y:y + height])[:3]
        return img_grid

    def show(self, image, title, **kwargs):
        """Consecutively draw image in a 3 × 4 grid.
        """
        if image is None:
            return
        self.plot_number += 1
        plt.subplot(3, 5, self.plot_number)
        plt.imshow(image, **kwargs)
        plt.title(title)

    def show_all(self, images):
        """Convenient function to call show for each item in a list.
        """
        for image in images:
            self.show(**image)
        plt.show()


class LinePattern(object):
    """Represents a repetitive line pattern, with `start` and `step`:

    `start` being where the pattern starts, and `step`, the space
    between each lines.
    """
    def __init__(self, start, step):
        self.start = start % step
        self.step = step

    def __str__(self):
        return "<LinePattern {} {}>".format(self.start, self.step)

    def coordinates_up_to(self, maximum):
        """Generator to give position of each line in this pattern up to a
        given maximum.
        """
        yield from ((x, self.step) for x in
                    range(self.start, maximum, self.step))


    @staticmethod
    def infer(positions, min_step=4):
        """Infer pattern from `positions`, a numpy array of values. returns
        the most evident repetitive pattern in the input.

        For a positions array like:
        [0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 100, ...
        returns LinePattern(start=5, step=5)

        This is done by brute force, trying each possibilities, and
        summing the values for each possibility. This returns only
        the best match.

        """
        positions = positions - positions.mean()
        positions = np.convolve(positions, (1 / 3, 2 / 3, 1 / 3))
        best = (0, 0, 0)
        for start, step, value in [
                (start, step, sum(positions[start::step]))
                for step in range(min_step, int(len(positions) / 2))
                for start in range(int(len(positions) / 2))]:
            if value > best[2]:
                best = start, step, value
        return LinePattern(best[0], best[1])


class Grid(object):
    """Given the result of a cv2.Canny, find a grid in the given image.

    The grid have no start and no end, only a "cell width" and an
    anchor (an intersection, so you can anlign the grid with the image).
    Exposes a list of columns (x, width) and a list of rows (y, height),
    as self.all_x and self.all_y.

    Exposes a all_cells() method, yielding every cells as tuples
    of (x, y, width, height).

    And a draw(self, image, color=(255, 0, 0), thickness=2) method,
    to draw the grid on a given image, usefull to check for correctness.

    """
    def __init__(self, edges):
        self.lines = lines = self.keep_lines(edges)
        self.columns = columns = self.keep_cols(edges)
        min_x_step = int(edges.shape[0] / 50)
        min_y_step = int(edges.shape[1] / 50)
        self.xpattern = LinePattern.infer(np.sum(lines, axis=1), min_x_step)
        self.ypattern = LinePattern.infer(np.sum(columns, axis=0), min_y_step)
        self.all_x = list(self.xpattern.coordinates_up_to(edges.shape[0]))
        self.all_y = list(self.ypattern.coordinates_up_to(edges.shape[1]))

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

    def cells_line_by_line(self):
        """
        Return all cells, line by line, like:
        [[(x, y), (x, y), ...]
         [(x, y), (x, y), ...]
         ... ]
        """
        for x, _ in self.all_x:
            yield [(x, y) for y, height in self.all_y]

    def all_cells(self):
        """
        returns tuples of (x, y, width, height) for each cell.
        """
        for x, width in self.all_x:
            for y, height in self.all_y:
                yield (x, y, width, height)

    def draw(self, image, color=(255, 0, 0), thickness=2):
        """Draw the current grid
        """
        for x, width in self.all_x:
            for y, height in self.all_y:
                cv2.rectangle(image, (y, x), (y + height, x + width),
                              color, thickness)

    def __str__(self):
        return '<Grid of {} lines, {} cols, cells: {}px × {}px>'.format(
            len(self.all_y), len(self.all_x),
            self.xpattern.step, self.ypattern.step)


def angle_between(line_a, line_b):
    """Compute the angle between two lines, in radians, in [0; π/2] line_a
    and line_b as tuples of x1, y1, x2, y2.

    Angle can only be in the range .

    >>> angle_between((0, 0, 10, 0), (0, 0, 0, 10))
    1.5707963267948966

    """
    distance = (abs(math.atan2(line_a[3] - line_a[1],
                               line_a[2] - line_a[0]) -
                    math.atan2(line_b[3] - line_b[1],
                               line_b[2] - line_b[0])) %
                (math.pi * 2))
    angle = distance - math.pi if distance >= math.pi / 2 else distance
    return abs(angle)


def find_orthogonal_lines(lines):
    """For a given set of lines as given by HoughLinesP, find four lines,
    such as the two first lines are "as perpendicular as possible" to
    the two second lines, therefore, forming a kind of rectangle.

    Returns a tuple of those four lines.
    >>> lines = find_orthogonal_lines([[[0, 1, 0, 0]],
    ...                                  [[0, 0, 1, 0]],
    ...                                  [[1, 0, 1, 1]],
    ...                                  [[1, 1, 0, 1]],
    ...                                  [[0, 0, 1, 1]]])
    >>> [0, 0, 1, 1] not in lines
    True
    """
    import sklearn.cluster
    kmeans = sklearn.cluster.KMeans(n_clusters=2)
    clusters = kmeans.fit_predict([[angle_between((0, 0, 0, 1), line[0])] for
                                   line in lines])
    bucket_a = []
    bucket_b = []
    for line_index, line in enumerate(lines):
        if clusters[line_index] == 0:
            bucket_a.append(line[0])
        else:
            bucket_b.append(line[0])
    if len(bucket_a) < 2 or len(bucket_b) < 2:
        raise GridFinderException("Not enough lines to find a grid.")
    bucket_a = sorted(bucket_a, key=line_length, reverse=True)[:6]
    bucket_b = sorted(bucket_b, key=line_length, reverse=True)[:6]
    bucket_a = sorted(bucket_a, key=lambda x: x[0])
    bucket_b = sorted(bucket_b, key=lambda x: x[2])
    return bucket_a[0], bucket_a[-1], bucket_b[0], bucket_b[-1]


def line_intersection(line1, line2):
    """Find the intersection point between two given lines.
    Lines given as typle of points: ((x1, y1), (x2, y2)).

    >>> line_intersection(((-10, 10), (10, -10)), ((-10, -10), (10, 10)))
    (0.0, 0.0)

    >>> line_intersection(((-10, 0), (10, 0)), ((0, -10), (0, 10)))
    (0.0, 0.0)
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    det = lambda x, y: x[0] * y[1] - x[1] * y[0]
    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Lines do not intersect
    something = (det(*line1), det(*line2))
    x = det(something, xdiff) / div
    y = det(something, ydiff) / div
    return x, y


def sort_points(points):
    """Sort points returning them in this order:
    top-left, top-right, bottom-right, bottom-left.
    Each poins is given as an (x, y) tuple.
    """
    from_top_to_bottom = sorted(points, key=lambda x: x[1])
    top = from_top_to_bottom[:2]
    bottom = from_top_to_bottom[2:]
    top_left = top[1] if top[0][0] > top[1][0] else top[0]
    top_right = top[0] if top[0][0] > top[1][0] else top[1]
    bottom_left = bottom[1] if bottom[0][0] > bottom[1][0] else bottom[0]
    bottom_right = bottom[0] if bottom[0][0] > bottom[1][0] else bottom[1]
    return top_left, top_right, bottom_right, bottom_left


def parse_args():
    """Return parsed arguments from command line"""
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Grid finder')
    parser.add_argument('file', help='Input image')
    parser.add_argument('--test', help='Run doctests', action='store_true')
    parser.add_argument('--debug',
                        help='Using matplotlib, display information '
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
    return parser.parse_args()


def line_length(line):
    """Mesure the given line as a (x1, y1, x2, y2) tuple.
    >>> line_length((0, 0, 0, 1))
    1.0
    >>> line_length((0, 0, 1, 1))
    1.4142135623730951
    >>> line_length((1, 1, 0, 0))
    1.4142135623730951
    """
    height = abs(line[1] - line[3])
    width = abs(line[0] - line[2])
    return math.sqrt(width ** 2 + height ** 2)


def find_lines(edges, min_line_length=200):
    """Find lines in Canny result, by using HoughLinesP.  Start with given
    `min_line_length`, and divide this minimum by two
    each time not enough lines are found to form a rectangle.

    returns the np array given by HoughLinesP, of the form:
    ```
    [[[122 209 428 127]]
     [[ 79 149 362  84]]
     [[ 24  94 311  39]]]
    ```
    """
    while min_line_length > 2:
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 40, np.array([]),
                                minLineLength=min_line_length, maxLineGap=10)
        if lines is not None:
            try:
                find_orthogonal_lines(lines)
            except GridFinderException:
                pass
            else:
                return lines
        min_line_length /= 2
    raise GridFinderException("No lines found")


def draw_lines(edges, lines):
    """Draw each lines in the given image.
    Lines given as an nparray typically given by HoughLinesP:
    [[[x1, y1, x2, y2]],
     [[x1, y1, x2, y2]]]
    """
    found_lines = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for start, end in [((lines[i][0][0], lines[i][0][1]),
                        (lines[i][0][2], lines[i][0][3])) for i in
                       range(lines.shape[0])]:
        cv2.line(found_lines, start, end, (0, 0, 255), 3, cv2.LINE_AA)
    return found_lines


def warp_image(image, top_left, top_right, bottom_right, bottom_left):
    """Warp the given image into the given box coordinates.
    """
    width = np.linalg.norm(np.array(top_left, np.float32) -
                           np.array(top_right, np.float32))
    height = np.linalg.norm(np.array(top_right, np.float32) -
                            np.array(bottom_right, np.float32))
    quad_pts = np.array([top_left,
                         (top_left[0] + width, top_left[1]),
                         (top_left[0] + width, top_left[1] + height),
                         (top_left[0], top_left[1] + height)], np.float32)
    trans = cv2.getPerspectiveTransform(
        np.array((top_left, top_right, bottom_right, bottom_left), np.float32),
        quad_pts)
    return cv2.warpPerspective(image, trans, (image.shape[1], image.shape[0]))


def print_grid_to_term(img, grid):
    """Print the given grid, in ascii, using 256 colors, to the terminal.
    """
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
            color = cv2.mean(img[x:x + grid.xpattern.step,
                                 y:y + grid.ypattern.step])
            print_color('[]', color=(color[:3]), end='')
        print()


def print_grid_as_json(img, grid):
    """Export the given grid as a json file containing a list of cells as:
    {'x': ..., 'y': ..., 'color': ...}
    """
    import json
    lines = []
    for line in grid.cells_line_by_line():
        line = [(x, y, cv2.mean(img[x:x + grid.xpattern.step,
                                    y:y + grid.ypattern.step]))
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


def write_grid_in_file(img, grid, imwrite):
    """Write given grid as a new image in the given file.
    """
    img_flat = img.copy()
    for x, y, width, height in grid.all_cells():
        mean_color = cv2.mean(img[x:x + width, y:y + height])[:3]
        img_flat[x:x + width, y:y + height] = mean_color
    cv2.imwrite(imwrite, img_flat)


def find_rectangle(best_lines):
    """Given a list of lines, return a tuple of four points, ordered in
    this order:
     - top_left
     - top_right
     - bottom_right
     - bottom_left

    >>> tl, tr, br, bl = find_rectangle(((122, 209, 428, 127),
    ...                                  (79, 149, 362, 84),
    ...                                  (151, 78, 297, 219),
    ...                                  (278, 56, 447, 174)))
    >>> tl
    (196.55904682849797, 121.99880549875489)
    >>> tr
    (328.96770995290564, 91.58692174226549)
    >>> br
    (393.08613857423046, 136.35600208141537)
    >>> bl
    (250.88330490946697, 174.46264378243043)
    """
    points = [line_intersection(((x1, x2), (y1, y2)), ((X1, X2), (Y1, Y2))) for
              (x1, x2, y1, y2), (X1, X2, Y1, Y2) in
              itertools.combinations(best_lines, 2)]
    points = [point for point in points if
              point is not None and point[0] > 0 and point[1] > 0]
    return sort_points(points)


def find_grid(filename, debug=False):
    """Find a grid pattern in the given file.

    Returns a tuple containing a flattened image so the found grid is
    now a rectangle, and a `Grid` object representing the found grid.
    """
    img = cv2.imread(filename)
    edges = cv2.Canny(img, 100, 200)  # try 66, 133, 3 ?
    lines = find_lines(edges)
    best_lines = find_orthogonal_lines(lines)
    top_left, top_right, bottom_right, bottom_left = find_rectangle(best_lines)
    warped = warp_image(img, top_left, top_right, bottom_right, bottom_left)
    warped_edges = cv2.Canny(warped, 100, 200)  # try 66, 133, 3 ?
    grid = Grid(warped_edges)
    if debug:
        points = (top_left, top_right, bottom_right, bottom_left)
        DebugUI().show_all([
            {"image": img, "title": 'Original image'},
            {"image": edges, "title": 'Edge Image', 'cmap': 'gray'},
            {"image": draw_lines(edges, lines),
             "title": "All lines", 'cmap': 'gray'},
            {"image": DebugUI.draw_interesting_lines(edges, best_lines, points),
             "title": 'Interesting lines',
             'cmap': 'gray'},
            {"image": DebugUI.draw_target_points(edges, points),
             "title": 'Target transformation points',
             'cmap': 'gray'},

            {"image": warped, "title": 'Warped', 'cmap': 'gray'},
            {"image": warped_edges, "title": 'Warped edges', 'cmap': 'gray'},
            {"image": grid.lines, "title": 'Lines', 'cmap': 'gray'},
            {"image": grid.columns, "title": 'Columns', 'cmap': 'gray'},
            {"image": DebugUI.draw_grid(warped, grid),
             "title": 'Detected {} lines, {} rows of {}px x {}px'.format(
                 len(grid.all_y), len(grid.all_x),
                 grid.xpattern.step, grid.ypattern.step), 'cmap': 'gray'},
            {"image": DebugUI.draw_flat_grid(warped, grid),
             "title": 'Reconstitution', 'cmap': 'gray'}])
    return warped, grid


def _main():
    """Run from command line, parsing command line arguments"""
    args = parse_args()
    if args.test:
        import doctest
        doctest.testmod()
        exit(0)
    img, grid = find_grid(args.file, args.debug)
    if args.term:
        print_grid_to_term(img, grid)
    if args.json:
        print_grid_as_json(img, grid)
    if args.imwrite:
        write_grid_in_file(img, grid, args.imwrite)
    if args.verbose:
        print('First column at:', grid.xpattern.start)
        print('First row at:', grid.ypattern.start)
        print('Column width:', grid.xpattern.step)
        print('Row width:', grid.ypattern.step)

if __name__ == '__main__':
    _main()
