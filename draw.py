import turtle as t
from math import ceil, cos, sin, radians
from svgpathtools import svg2paths2
import numpy as np
import re


def read_svg(path="bacho.svg", seg_unit=8):
    paths, attrs, svg_attr = svg2paths2(path)

    def _to_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(str(value).replace('px', '').strip())
        except Exception:
            return default

    width = _to_float(svg_attr.get('width'))
    height = _to_float(svg_attr.get('height'))
    svg_size = (width, height)

    vb_attr = svg_attr.get('viewBox') or svg_attr.get('viewbox')
    if vb_attr:
        parts = vb_attr.replace(',', ' ').split()
        if len(parts) == 4:
            viewbox = [float(f) for f in parts]
        else:
            viewbox = [0.0, 0.0, width, height]
    else:
        # With explicit width/height but no viewBox, use [0,0,width,height]
        if width > 0 and height > 0:
            viewbox = [0.0, 0.0, width, height]
        else:
            # Fallback: compute from path bounds
            min_x = float('inf')
            min_y = float('inf')
            max_x = float('-inf')
            max_y = float('-inf')
            for path in paths:
                try:
                    x0, x1, y0, y1 = path.bbox()
                    min_x = min(min_x, x0)
                    max_x = max(max_x, x1)
                    min_y = min(min_y, y0)
                    max_y = max(max_y, y1)
                except Exception:
                    continue
            if min_x == float('inf') or min_y == float('inf') or max_x == float('-inf') or max_y == float('-inf'):
                viewbox = [0.0, 0.0, 1000.0, 1000.0]
            else:
                viewbox = [min_x, min_y, max_x, max_y]

    # --- Transform utilities ---
    def mat_identity():
        return [[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]]

    def mat_mul(a, b):
        return [
            [a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0],
             a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1],
             a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2]],
            [a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0],
             a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1],
             a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2]],
            [a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0],
             a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1],
             a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2]],
        ]

    def mat_apply(m, x, y):
        nx = m[0][0]*x + m[0][1]*y + m[0][2]
        ny = m[1][0]*x + m[1][1]*y + m[1][2]
        return nx, ny

    float_re = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    def parse_numbers(s):
        return [float(v) for v in re.findall(float_re, s)]

    def parse_transform(s):
        if not s:
            return mat_identity()
        s = s.strip()
        # Apply left-to-right as in SVG spec (list composes as Tn * ... * T1)
        m = mat_identity()
        for func, args_str in re.findall(r"(\w+)\s*\(([^)]*)\)", s):
            vals = parse_numbers(args_str)
            name = func.lower()
            tm = mat_identity()
            if name == 'translate':
                tx = vals[0] if len(vals) > 0 else 0.0
                ty = vals[1] if len(vals) > 1 else 0.0
                tm = [[1,0,tx],[0,1,ty],[0,0,1]]
            elif name == 'scale':
                sx = vals[0] if len(vals) > 0 else 1.0
                sy = vals[1] if len(vals) > 1 else sx
                tm = [[sx,0,0],[0,sy,0],[0,0,1]]
            elif name == 'rotate':
                ang = vals[0] if len(vals) > 0 else 0.0
                cx = vals[1] if len(vals) > 2 else 0.0
                cy = vals[2] if len(vals) > 2 else 0.0
                a = cos(radians(ang)); b = sin(radians(ang))
                if len(vals) > 2:
                    # Translate to origin, rotate, translate back
                    tm = mat_mul(mat_mul([[1,0,cx],[0,1,cy],[0,0,1]], [[a,-b,0],[b,a,0],[0,0,1]]), [[1,0,-cx],[0,1,-cy],[0,0,1]])
                else:
                    tm = [[a,-b,0],[b,a,0],[0,0,1]]
            elif name == 'matrix' and len(vals) >= 6:
                a,b,c,d,e,f = vals[:6]
                tm = [[a,c,e],[b,d,f],[0,0,1]]
            # Note: skewX/skewY not handled explicitly
            m = mat_mul(tm, m)
        return m

    # Build polygons with per-path transforms applied
    transformed_polys = []
    new_attrs = []
    for path, attr in zip(paths, attrs):
        # Inline style -> expand
        if 'style' in attr:
            for item in str(attr['style']).split(';'):
                if not item:
                    continue
                if ':' in item:
                    k, v = item.split(':', 1)
                    attr[k.strip()] = v.strip()
        # Default stroke
        if 'stroke' not in attr and 'fill' in attr:
            attr['stroke'] = attr['fill']

        m = parse_transform(attr.get('transform', ''))
        poly = []
        for subpaths in path.continuous_subpaths():
            points = []
            for seg in subpaths:
                interp_num = max(2, ceil(seg.length()/seg_unit))
                points.append(seg.point(np.linspace(0, 1, interp_num, endpoint=True)))
            points = np.concatenate(points)
            points = np.append(points, points[0])
            # Apply transform
            ring = []
            for p in points:
                x, y = p.real, p.imag
                x, y = mat_apply(m, x, y)
                ring.append((x, y))
            poly.append(ring)
        transformed_polys.append(poly)
        new_attrs.append(attr)

    return (transformed_polys, new_attrs, svg_size, viewbox)



def head_to(t, x, y, draw=True, have_sprite=True):
    wasdown = t.isdown()
    heading = t.towards(x,y)
    t.pen(pendown=draw)
    t.seth(heading)
    t.clearstamps()
    t.goto(x,y)
    t.stamp()
    t.pen(pendown=wasdown)



def draw_polygon(t, poly, fill='black', stroke='black', have_sprite=True):
    if fill=='none':
        fill = 'black'
    t.color(stroke,fill)
    p = poly[0]
    head_to(t,p[0],-(p[1]), False, have_sprite)
    for p in poly[1:]: 
        head_to(t,p[0],-(p[1]), have_sprite=have_sprite)
    t.up()


def draw_multipolygon(t, mpoly, fill='black', stroke='black', have_sprite=True):
    p = mpoly[0][0]
    head_to(t,p[0],-(p[1]), False, have_sprite)
    if fill!='none':
        t.begin_fill()
    for i, poly in enumerate(mpoly):
        draw_polygon(t, poly, fill, stroke, have_sprite)
        if i!=0:
            head_to(t,p[0],-(p[1]), False, have_sprite)
    if fill!="none":
        t.end_fill()


def compute_bounds(polys):
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    for mpoly in polys:
        for ring in mpoly:
            for x, y in ring:
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y
    if min_x == float('inf'):
        return [0.0, 0.0, 1000.0, 1000.0]
    return [min_x, min_y, max_x, max_y]


def main():
    polys, attrs, svg_size, viewbox = read_svg()

    # Prefer original width/height if provided to preserve aspect/position
    width, height = svg_size
    if width > 0 and height > 0:
        vb = [0.0, 0.0, width, height]
    else:
        vb = compute_bounds(polys)

    vb_w = max(vb[2]-vb[0], 1e-6)
    vb_h = max(vb[3]-vb[1], 1e-6)
    ar = vb_w/vb_h

    window = t.Screen()
    win_m = min(window.window_width(),window.window_height())
    if ar>1:
        window.setup(win_m*ar, win_m)
    else:
        window.setup(win_m, win_m/ar)

    t.reset()
    t.speed(0)

    margin = 0.02
    dx = vb_w * margin
    dy = vb_h * margin
    llx = (vb[0] - dx)
    lly = -(vb[3] + dy)
    urx = (vb[2] + dx)
    ury = -(vb[1] - dy)
    t.setworldcoordinates(llx, lly, urx, ury)

    t.mode(mode='world')
    t.tracer(n=10, delay=0)

    # Draw in original document order
    for poly, attr in zip(polys, attrs):
        t.pen(outline=0.5)
        if 'stroke-width' in attr:
            try:
                t.pen(outline=float(attr['stroke-width']), pencolor='black')  # type: ignore
            except Exception:
                pass

        fill = attr.get('fill', 'none')
        stroke = attr.get('stroke', 'black')
        draw_multipolygon(t, poly, fill=fill, stroke=stroke)

    t.tracer(n=1, delay=0)
    t.clearstamps()
    t.penup()


if __name__ == '__main__':
    main()


