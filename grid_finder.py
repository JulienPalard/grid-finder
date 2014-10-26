#!/usr/bin/env python3

import cv2
import numpy as np


class Plotting(object):
    """
    Just for debug purposes, call it, max 6 times,
    to add a plot to a multiple plot.
    like:
    with Plotting():
        plt.plot(...)
    """
    plot_number = 1

    def __enter__(self):
        from matplotlib import pyplot as plt
        plt.subplot(3, 4, Plotting.plot_number)
        Plotting.plot_number += 1
        return plt

    def __exit__(self, type, value, traceback):
        from matplotlib import pyplot as plt
        plt.xticks([])
        plt.yticks([])


def show_debug(img, grid, edges):
    img_grid = img.copy()
    grid.draw(img_grid, (255, 255, 255))

    img_grid_flat_color = img_grid.copy()
    for x, y, width, height in grid.all_cells():
        mean_color = cv2.mean(img[x:x + width, y:y + height])[:3]
        img_grid_flat_color[x:x + width, y:y + height] = mean_color

    with Plotting() as plt:
        plt.imshow(img, cmap = 'gray')
        plt.title('Original Image')

    with Plotting() as plt:
        plt.imshow(edges, cmap = 'gray')
        plt.title('Edge Image')

    with Plotting() as plt:
        plt.imshow(img_grid, cmap = 'gray')
        plt.title('Detected {} lines, {} rows of {}px x {}px'.
                  format(len(grid.all_y), len(grid.all_x),
                         grid.xstep, grid.ystep)),

    with Plotting() as plt:
        plt.imshow(img_grid_flat_color, cmap = 'gray')
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
        lines = self.keep_lines(edges)
        columns = self.keep_cols(edges)

        if debug:
            with Plotting() as plt:
                plt.imshow(lines, cmap = 'gray')
                plt.title('Lines')

        if debug:
            with Plotting() as plt:
                plt.imshow(columns, cmap = 'gray')
                plt.title('Columns')
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
        if debug:
            with Plotting() as plt:
                plt.plot(positions)
                plt.title(debug_title)
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
    args = parser.parse_args()

    img = cv2.imread(args.file)
    edges = cv2.Canny(img, 100, 200)
    grid = Grid(edges, debug=args.debug)
    if args.debug:
        show_debug(img, grid, edges)
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
    if args.verbose:
        print('First column at:', grid.x)
        print('First row at:', grid.y)
        print('Column width:', grid.xstep)
        print('Row width:', grid.ystep)
    sys.exit(0)
    print(grid.x, grid.y, grid.xstep, grid.ystep)

if __name__ == '__main__':
    main()
