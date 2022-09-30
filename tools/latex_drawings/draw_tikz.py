# reproduction of Figure 7 from http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf

import sane_tikz.core as stz
import sane_tikz.formatting as fmt
from example_matrices import nnz_locations_20x20

box_spacing = 1.1

lst = ["C", "A", "B"]
dashed = fmt.line_style("dashed")
regular_lw = fmt.line_width(fmt.standard_line_width)

c_coord = stz.center_coords
move = stz.translate_to_coords
a_circle_radius = 0.05

M = 10
K = 10 
N = 10


def coloured_point(colour='black', loc=[0,0]):
    if type(colour) == list:
        cricle_arcs = []
        num_parts = len(colour)
        p_angle = 360 / num_parts
        for part in range(num_parts):
            f_fmt = fmt.line_and_fill_colors(colour[part], colour[part])
            if part > 0:
                loc = stz.translate_coords_horizontally(loc, -2*a_circle_radius)

            cricle_arcs.append(stz.circular_arc(
                loc, a_circle_radius, 
                part*p_angle, (part + 1) * p_angle, f_fmt))
        return cricle_arcs

    else:
        f_fmt = fmt.line_and_fill_colors(colour, colour)
        return [stz.circle(loc, a_circle_radius, f_fmt)]


def box(box_width, box_height, fmt=dashed):
    return [stz.rectangle([0, 0], [box_width, -box_height], fmt)]


def _lines_to_draw(n, only_draw):
    if only_draw is None: return range(0,n)
    if type(only_draw) is int: return range(0, only_draw+1)
    if type(only_draw) is list: return only_draw
    raise NotImplemented()


def partition_vert(box, n, only_draw=None, line_style=dashed, box_style=None, width=1, offset=0):
    top_left, bottom_right = stz.bbox(box)
    top_right = [bottom_right[0], top_left[1]]

    y_dist = stz.y_difference(stz.bbox(box))
    x_dist = stz.x_difference(stz.bbox(box))
    p_dist = x_dist / n

    if box_style is not None:
        boxes = []
        for i in _lines_to_draw(n, only_draw):
            print(offset, offset*x_dist)
            corner = stz.translate_coords_vertically(top_left, (-i) * p_dist)
            corner = stz.translate_coords_horizontally(corner, offset*x_dist)
            boxes += [stz.rectangle(
                corner,
                stz.translate_coords(corner, width*x_dist, -p_dist),
                box_style
            )]
        return boxes
    else:
        return [stz.line_segment(
            stz.translate_coords_vertically(top_left, -i * p_dist),
            stz.translate_coords_vertically(top_right, -i * p_dist),
            line_style
        ) for i in _lines_to_draw(n, only_draw)]


def partition_horz(box, n, only_draw=None, line_style=dashed, box_style=None):
    top_left, bottom_right = stz.bbox(box)
    bot_left = [top_left[0], bottom_right[1]]

    y_dist = stz.y_difference(stz.bbox(box))
    x_dist = stz.x_difference(stz.bbox(box))
    p_dist = y_dist / n

    if box_style is not None:
        return [stz.rectangle(
            stz.translate_coords_horizontally(top_left, -(i+1) * p_dist),
            [-p_dist, x_dist],
            box_style
        ) for i in _lines_to_draw(n, only_draw)]
    else:
        return [stz.line_segment(
            stz.translate_coords_horizontally(top_left, -i * p_dist),
            stz.translate_coords_horizontally(bot_left, -i * p_dist),
            line_style
        ) for i in _lines_to_draw(n, only_draw)]


def partition_grid(box, hn, vn=None, labels=None):
    if vn is None: vn = hn
    ret = []
    ret += partition_vert(box, hn)
    ret += partition_horz(box, vn)

    if labels is not None:
        top_left, bottom_right = stz.bbox(box)
        y_dist = stz.y_difference(stz.bbox(box))
        x_dist = stz.y_difference(stz.bbox(box))
        yp_dist = y_dist / vn
        xp_dist = x_dist / hn

        assert len(labels) == vn*hn

        shift_right = lambda c, x: stz.translate_coords_horizontally(c, -x)
        shift_down  = lambda c, x: stz.translate_coords_vertically(c, x)

        base_coord = shift_right(top_left, xp_dist / 2)
        base_coord = shift_down(base_coord, yp_dist / 2)

        for i in range(vn*hn):
            x = i % hn
            y = i // vn

            text_coords = shift_right(base_coord, x * xp_dist)
            text_coords = shift_down(text_coords, y * yp_dist)

            ret += [stz.latex(text_coords, labels[i])]

    return ret


def add_text_below(box, s):
    top_left, bottom_right = stz.bbox(box)
    x_dist = stz.x_difference(stz.bbox(box))
    bot_center = [top_left[0] + x_dist / 2, bottom_right[1]]
    loc = stz.translate_coords_vertically(bot_center, -0.3)

    return [stz.latex(loc, s)]


def place_nnzs_in(box, colour_nnz=None, colour_all=['blue', 'red'], colour_specific=None):
    top_left, bottom_right = stz.bbox(box)
    y_dist = stz.y_difference(stz.bbox(box))
    x_dist = stz.y_difference(stz.bbox(box))

    points = []
    current_colour_idx = 0
    for point in nnz_locations_20x20:
        _colour_nnz = False

        if colour_nnz is not None:
            if type(colour_nnz) == list:
                for coloured_nnz in colour_nnz:
                    if all(coloured_nnz == point): _colour_nnz = True
            else:
                if (point[0] < colour_nnz):
                    _colour_nnz = True

        point = point * 0.05 + 0.025

        if _colour_nnz:
            if colour_specific is not None:
                colour = colour_specific[current_colour_idx]
                current_colour_idx += 1
            else:
                colour = colour_all

        loc = stz.translate_coords(top_left, -point[1] * x_dist, point[0] * y_dist)
        points += coloured_point(loc=loc, colour='black' if not _colour_nnz else colour)
    
    return points


##
#   Common
##

boxes = [box(5, 5, fmt=regular_lw) for _ in lst]
stz.distribute_horizontally_with_spacing(boxes, box_spacing)

eq = stz.latex([0, 0], "\\textbf{=}")
mul = stz.latex([0, 0], "\\textbf{x}")

move(eq,  c_coord(eq),  c_coord(boxes[0:2]))
move(mul, c_coord(mul), c_coord(boxes[1:3]))

num_paritions = 20

##
#   CSR_C 2D
##

partitions  = partition_vert(boxes[0], num_paritions, only_draw=2,
                             box_style=fmt.line_and_fill_colors('black', 'blue'),
                             width=0.25)
partitions += partition_vert(boxes[0], num_paritions, only_draw=2,
                             box_style=fmt.line_and_fill_colors('black', 'red'),
                             width=0.25, offset=0.25)


partitions += partition_vert(boxes[1], num_paritions, only_draw=[3])

partitions += partition_vert(boxes[2], num_paritions, only_draw=[
                                0, 5, 6, 7, 8, 9, 10, 11, 13, 15],
                             box_style=fmt.line_and_fill_colors('black', 'blue'),
                             width=0.25, offset=0)
partitions += partition_vert(boxes[2], num_paritions, only_draw=[
                                0, 5, 6, 7, 8, 9, 10, 11, 13, 15],
                             box_style=fmt.line_and_fill_colors('black', 'red'),
                             width=0.25, offset=0.25)


text  = add_text_below(boxes[0], "C")
text += add_text_below(boxes[1], "A")
text += add_text_below(boxes[2], "B")

points = place_nnzs_in(boxes[1], colour_nnz=3)

stz.draw_to_tikz_standalone([boxes, eq, mul] + partitions + text + points, "tex/csr_c_2D.tex")

##
#   CSR_C 1D
##

partitions  = partition_vert(boxes[0], num_paritions, only_draw=0,
                             box_style=fmt.line_and_fill_colors('black', 'blue'))
partitions += partition_vert(boxes[0], num_paritions, only_draw=1,
                             box_style=fmt.line_and_fill_colors('black', 'red'))


partitions += partition_vert(boxes[1], num_paritions, only_draw=[1, 2])


partitions += partition_vert(boxes[2], num_paritions, only_draw=[7, 8, 10],
                             box_style=fmt.line_and_fill_colors('black', 'blue'),
                             width=0.5)
partitions += partition_vert(boxes[2], num_paritions, only_draw=[7, 8, 10],
                             box_style=fmt.line_and_fill_colors('black', 'red'),
                             width=0.5, offset=0.5)

text  = add_text_below(boxes[0], "C")
text += add_text_below(boxes[1], "A")
text += add_text_below(boxes[2], "B")

points = place_nnzs_in(boxes[1], colour_nnz=2, colour_specific=[
    'red', 'red', 'red',
    'blue', 'blue', 'blue', 'blue', 'blue'
])

stz.draw_to_tikz_standalone([boxes, eq, mul] + partitions + text + points, "tex/csr_c_1D.tex")

##
#   CSR_A
##

partitions  = partition_vert(boxes[0], num_paritions, only_draw=0,
                             box_style=fmt.line_and_fill_colors('red', 'blue'))

partitions += partition_vert(boxes[1], num_paritions, only_draw=[1])

partitions += partition_vert(boxes[2], num_paritions, only_draw=[7],
                             box_style=fmt.line_and_fill_colors('black', 'blue'))
partitions += partition_vert(boxes[2], num_paritions, only_draw=[8],
                             box_style=fmt.line_and_fill_colors('black', 'red'))

text  = add_text_below(boxes[0], "C")
text += add_text_below(boxes[1], "A")
text += add_text_below(boxes[2], "B")

points = place_nnzs_in(boxes[1], colour_nnz=[[ 0,  7], [ 0,  8]], colour_specific=['blue', 'red'])

stz.draw_to_tikz_standalone([boxes, eq, mul] + partitions + text + points, "tex/csr_a.tex")

##
#   CSR_C 2D B
##

partitions  = partition_vert(boxes[0], num_paritions, only_draw=2,
                             box_style=fmt.line_and_fill_colors('black', 'blue'),
                             width=0.25)
partitions += partition_vert(boxes[0], num_paritions, only_draw=2,
                             box_style=fmt.line_and_fill_colors('black', 'red'),
                             width=0.25, offset=0.25)


partitions += partition_vert(boxes[1], num_paritions, only_draw=[3])
partitions += partition_horz(boxes[1], int(num_paritions / 5))


partitions += partition_vert(boxes[2], num_paritions, only_draw=[
                                0, 5, 6, 7, 8, 9, 10, 11, 13, 15],
                             box_style=fmt.line_and_fill_colors('black', 'blue'),
                             width=0.25, offset=0)
partitions += partition_vert(boxes[2], num_paritions, only_draw=[
                                0, 5, 6, 7, 8, 9, 10, 11, 13, 15],
                             box_style=fmt.line_and_fill_colors('black', 'red'),
                             width=0.25, offset=0.25)


text  = add_text_below(boxes[0], "C")
text += add_text_below(boxes[1], "A")
text += add_text_below(boxes[2], "B")

points = place_nnzs_in(boxes[1], colour_nnz=3)

stz.draw_to_tikz_standalone([boxes, eq, mul] + partitions + text + points, "tex/csr_c_2D_b.tex")


##
#   CSR_C 2D C
##

partitions  = partition_vert(boxes[0], num_paritions, only_draw=3,
                             box_style=fmt.line_and_fill_colors('red', 'blue'))

partitions += partition_vert(boxes[1], num_paritions, only_draw=[3])
partitions += partition_horz(boxes[1], int(num_paritions / 5))

partitions += partition_vert(boxes[2], num_paritions, only_draw=[0, 5],
                             box_style=fmt.line_and_fill_colors('black', 'blue'))
partitions += partition_vert(boxes[2], num_paritions, only_draw=[6, 7, 8, 9, 10],
                             box_style=fmt.line_and_fill_colors('black', 'red'))


text  = add_text_below(boxes[0], "C")
text += add_text_below(boxes[1], "A")
text += add_text_below(boxes[2], "B")

points = place_nnzs_in(boxes[1], colour_nnz=3, colour_specific=[
    'red', 'red', 'black',
    'blue', 'red', 'red', 'black', 'black',
    'blue', 'red', 'red', 'black', 'black', 'black', 'black'
])

stz.draw_to_tikz_standalone([boxes, eq, mul] + partitions + text + points, "tex/csr_c_2D_c.tex")
