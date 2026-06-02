"""Simple drawings of objects."""

import numpy as np
from manim import *


class Clock(VGroup):
    def __init__(self, radius, stroke_width=6, **kwargs):
        super().__init__(**kwargs)
        clock_face = Circle(radius=radius, color=BLACK)
        clock_face.set_stroke(width=stroke_width, color=BLACK)
        center = Dot(radius=0.05 * radius, color=BLACK)

        hour_hand = Line(
            start=ORIGIN, end=0.6 * UP * radius, stroke_width=stroke_width, color=BLACK
        ).rotate(-PI / 5, about_point=ORIGIN)

        minute_hand = Line(
            start=ORIGIN, end=0.9 * UP * radius, stroke_width=stroke_width, color=BLACK
        ).rotate(-PI / 3, about_point=ORIGIN)

        clock = VGroup(clock_face, hour_hand, minute_hand, center)
        self.add(clock)


class Globe(VGroup):
    def __init__(self, radius, stroke_width=6, **kwargs):
        super().__init__(**kwargs)
        globe_outline = Circle(radius=radius, stroke_width=stroke_width, color=BLACK)
        # Vertical meridian
        meridian = Ellipse(
            width=2 * radius * 0.4,
            height=2 * radius,
            stroke_width=stroke_width,
            color=BLACK,
        )
        equator = Line(
            start=LEFT * (radius),
            end=RIGHT * (radius),
            stroke_width=stroke_width,
            color=BLACK,
        )

        arc_up = Circle(radius=radius, stroke_width=stroke_width, color=BLACK).shift(
            1.4 * UP * radius
        )
        arc_up = Intersection(
            globe_outline, arc_up, color=BLACK, stroke_width=stroke_width
        )
        arc_down = Circle(radius=radius, stroke_width=stroke_width, color=BLACK).shift(
            1.4 * DOWN * radius
        )
        arc_down = Intersection(
            globe_outline, arc_down, color=BLACK, stroke_width=stroke_width
        )

        globe = VGroup(globe_outline, meridian, equator, arc_up, arc_down)

        self.add(globe)


class Moon(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        big_circle = Circle(radius=1, color=BLACK, fill_opacity=1)
        small_circle = Circle(radius=1, color=BLACK, fill_opacity=1).shift(RIGHT * 0.5)

        crescent_moon = Difference(big_circle, small_circle, color=BLACK).set_fill(
            opacity=1, color=BLACK
        )
        self.add(crescent_moon)


class Cloud(VGroup):
    def __init__(self, cloud_type, stroke_w=6, **kwargs):
        super().__init__(**kwargs)

        if cloud_type == 1:
            circles = [
                Circle(radius=0.4).shift(0.5 * RIGHT + UP * 0.1),
                Circle(radius=0.5).shift(0.75 * RIGHT + DOWN * 0.25),
                Circle(radius=0.5).shift(0.75 * LEFT + DOWN * 0.25),
                Circle(radius=0.5).shift(0.5 * LEFT + UP * 0.3),
                Circle(radius=0.7).shift(DOWN * 0.2),
                Circle(radius=0.5).shift(UP * 0.2),
            ]

        elif cloud_type == 2:
            circles = [
                Circle(radius=0.45).shift(0.55 * RIGHT + UP * 0.1),
                Circle(radius=0.3).shift(1 * RIGHT + DOWN * 0.3),
                Circle(radius=0.4).shift(0.75 * LEFT + DOWN * 0.25),
                Circle(radius=0.2).shift(0.5 * LEFT + UP * 0.3),
                Circle(radius=0.5).shift(DOWN * 0.2),
                Circle(radius=0.35).shift(UP * 0.3),
                Circle(radius=0.3).shift(0.5 * LEFT + UP * 0.2),
                Circle(radius=0.3).shift(0.65 * RIGHT + DOWN * 0.4),
            ]

        else:
            raise ValueError("tree_type must be 1 or 2")

        cloud = Union(*circles).set_stroke(color=BLACK, width=stroke_w)

        self.add(cloud)


class Car(VGroup):
    def __init__(self, stroke_w=6, wheel_radius=0.3, **kwargs):
        super().__init__(**kwargs)

        # car body
        points = [
            [-1.75, -0.4, 0],
            [-1.6, 0.2, 0],
            [-1.1, 0.4, 0],
            [-0.8, 0.8, 0],
            [0.4, 0.8, 0],
            [0.5, 0.7, 0],
            [0.7, 0.45, 0],
            [1.6, 0.2, 0],
            [1.75, -0.4, 0],
        ]

        car_body = Polygon(*points, color=BLACK, fill_opacity=0).set_stroke(
            width=stroke_w
        )
        car_body.round_corners(0.15)

        # wheels
        wheel_offset_y = -0.75
        wheel_left = Arc(
            radius=wheel_radius,
            start_angle=0,
            angle=-PI,
            color=BLACK,
            stroke_width=stroke_w,
        )
        wheel_right = Arc(
            radius=wheel_radius,
            start_angle=0,
            angle=-PI,
            color=BLACK,
            stroke_width=stroke_w,
        )

        # place wheels below body
        wheel_left.move_to(car_body.get_left() + np.array([0.8, wheel_offset_y, 0]))
        wheel_right.move_to(car_body.get_right() + np.array([-0.8, wheel_offset_y, 0]))

        # group
        self.add(car_body, wheel_left, wheel_right)
        self.move_to(ORIGIN)


class Tree(VGroup):
    def __init__(
        self, tree_type=1, branch=False, crown_color=None, color=BLACK, **kwargs
    ):
        super().__init__(**kwargs)

        # Trunk
        top_width = 0.1
        bottom_width = 0.2
        height = 2
        radius = 0.05
        trapezoid = Polygon(
            [-top_width / 2, height / 2, 0],
            [top_width / 2, height / 2, 0],
            [bottom_width / 2, -height / 2, 0],
            [-bottom_width / 2, -height / 2, 0],
            color=color,
            fill_opacity=1,
        ).set_stroke(width=2)
        trapezoid.round_corners(radius)
        trapezoid.shift(DOWN * 1)

        if branch == True:
            trapezoid2 = Polygon(
                [-top_width / 4, height / 4, 0],
                [top_width / 4, height / 4, 0],
                [bottom_width / 4, -height / 4, 0],
                [-bottom_width / 4, -height / 4, 0],
                color=color,
                fill_opacity=1,
            ).set_stroke(width=2)
            trapezoid2.round_corners(radius)
            trapezoid2.shift(DOWN * 0.5 + 0.3 * RIGHT).rotate(-40 * DEGREES)

        # Crown (2 options)
        if tree_type == 1:
            c1 = Circle(radius=0.5).shift(0.25 * RIGHT + UP * 0.25)
            c2 = Circle(radius=0.5).shift(0.5 * RIGHT + DOWN * 0.25)
            c3 = Circle(radius=0.5).shift(0.5 * LEFT + DOWN * 0.25)
            c4 = Circle(radius=0.5).shift(0.3 * LEFT + UP * 0.25)
            c5 = Circle(radius=0.5).shift(DOWN * 0.5)

            crown = Union(c1, c2, c3, c4, c5).set_stroke(color=color, width=8)

        elif tree_type == 2:
            c1 = Circle(radius=0.5).shift(0.4 * RIGHT + UP * 0.25)
            c2 = Circle(radius=0.6).shift(0.5 * RIGHT + DOWN * 0.3)
            c3 = Circle(radius=0.7).shift(0.5 * LEFT + DOWN * 0.25)
            c4 = Circle(radius=0.5).shift(0 * LEFT + UP * 0.4)

            crown = Union(c1, c2, c3, c4).set_stroke(color=color, width=8)

        else:
            raise ValueError("tree_type must be 1 or 2")

        if crown_color:
            crown.set_fill(color=crown_color, opacity=0.2)

        if branch == True:
            self.tree = VGroup(crown, trapezoid2, trapezoid)
        else:
            self.tree = VGroup(crown, trapezoid)
        self.add(self.tree)

    def place_tree(self, axes, ground, x_axis, scale=1):
        x_scene = axes.c2p(x_axis, 0)[0]

        def tree_pos():
            t_copy = self.tree.copy()

            # get scale value (float or ValueTracker)
            s = scale.get_value() if isinstance(scale, ValueTracker) else scale

            # move bottom to origin, then scale
            t_copy.scale(s)
            bottom = t_copy.get_bottom()
            t_copy.shift(-bottom)  # bottom at origin

            # interpolate y along ground
            ground_points = ground.get_points()[:, :2]
            closest_idx = np.argmin(np.abs(ground_points[:, 0] - x_scene))
            y_ground = ground_points[closest_idx, 1]

            # move tree so bottom is at ground
            t_copy.shift(np.array([x_scene, y_ground, 0]))

            return t_copy

        return always_redraw(tree_pos)
