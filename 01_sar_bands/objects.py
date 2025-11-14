from manim import *
import numpy as np
import random


class Cave(VGroup):
    def __init__(self, radius, color=GREY, fill_opacity=0.5, num_points=20, **kwargs):
        super().__init__(**kwargs)
        points = []
        for i in range(num_points):
            angle = i * TAU / num_points
            # vary radius for organic shape
            r = radius * (0.8 + 0.4 * np.random.rand())
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append(np.array([x, y, 0]))

        blob = VMobject(color=color, fill_opacity=fill_opacity)
        blob.set_points_as_corners(points + [points[0]])
        blob.set_smooth(True)

        self.add(blob)


class Dune(VGroup):
    def __init__(
        self,
        amplitude,
        wavelength,
        shift,
        height_shift,
        extent,
        extent_dots,
        y_bottom_dots,
        **kwargs
    ):
        super().__init__(**kwargs)

        dune = self.dune_curve(
            amplitude, wavelength, shift, height_shift, extent, color=BLACK
        )
        dune_f = self.dune_function(amplitude, wavelength, shift, height_shift)
        dots = self.fill_with_dots(
            f=dune_f,
            n_dots=60,
            color=BLACK,
            x_min=extent_dots[0],
            x_max=extent_dots[1],
            y_bottom=y_bottom_dots,
        )

        self.add(dune, dots)

    @staticmethod
    def dune_curve(amplitude, wavelength, shift, height_shift, extent, color=BLACK):
        return ParametricFunction(
            lambda t: np.array(
                [t, amplitude * np.sin((t + shift) /
                                       wavelength) + height_shift, 0]
            ),
            t_range=extent,
            color=color,
        ).set_stroke(width=6)

    @staticmethod
    def dune_function(amplitude, wavelength, shift, height_shift):
        """Function to sample y-values for the dune."""
        return lambda x: amplitude * np.sin((x + shift) / wavelength) + height_shift

    @staticmethod
    def fill_with_dots(
        f, n_dots=100, color=BLACK, x_min=-4, x_max=4, y_bottom=-4, radius=0.03
    ):
        """Generate random dots under a curve that look like sand."""
        dots = VGroup()
        for _ in range(n_dots):
            x = random.uniform(x_min, x_max)
            y_top = f(x)
            y = random.uniform(y_bottom, y_top)
            # slight jitter so dots don't look perfectly grid-like
            jitter_x = random.uniform(-0.02, 0.02)
            jitter_y = random.uniform(-0.02, 0.02)
            d = Dot(point=[x + jitter_x, y + jitter_y, 0],
                    radius=radius, color=color)
            dots.add(d)
        return dots


class Sat(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        bottom_rect = Rectangle(
            width=2,
            height=0.1,
            color=BLACK,
            fill_opacity=1,
        )
        top_rect = Rectangle(
            width=1.0,
            height=0.5,
            color=BLACK,
            fill_opacity=1,
        )
        top_rect.next_to(bottom_rect, UP, buff=0).align_to(
            bottom_rect, RIGHT + LEFT)

        # Group them
        sat = VGroup(bottom_rect, top_rect)
        self.add(sat)

    def send_beam(
        self,
        sat_initial_pos,
        axes,
        target_point=(-0.5, -1.5),
        travel_direction="forward",
        amplitude=0.25,
        frequency=1 / 0.5,
        x_range=[0, 1],
    ):
        """Produces an animation to send a wave with a set amplitude and frequency from sat_initial_pos to target_point."""

        start_wave = sat_initial_pos
        target_pos = axes.c2p(*target_point)
        direction = target_pos - start_wave
        direction_angle = np.arctan2(direction[1], direction[0])
        if travel_direction == "forward":
            wave_direction = -1
        elif travel_direction == "backward":
            wave_direction = 1

        # trackers
        t_tracker = ValueTracker(0)
        pos_tracker = ValueTracker(0)

        def get_wave_pos():
            return start_wave + pos_tracker.get_value() * direction

        wave = always_redraw(
            lambda: FunctionGraph(
                lambda x: amplitude
                * np.sin(
                    2
                    * np.pi
                    * frequency
                    * (x + wave_direction * 0.5 * t_tracker.get_value())
                ),
                x_range=x_range,
                color=BLACK,
                stroke_width=5,
            )
            .rotate(direction_angle - np.pi / 2, about_point=get_wave_pos())
            .move_to(get_wave_pos())
        )
        wave.add_updater(
            lambda m: m.rotate(direction_angle, about_point=get_wave_pos())
        )

        # build animations (but not play yet)
        animations = [t_tracker.animate.set_value(
            3), pos_tracker.animate.set_value(1)]

        return wave, animations

    def send_beam_multipath(
        self, path_points, amplitude=0.25, frequency=1 / 0.5, travel_direction="forward"
    ):
        """Produces an animation to send a wave with set amplitude and frequency
        along a path consisting of several points (provided as a list in path.points).
        """

        if travel_direction == "forward":
            wave_direction = -1
        elif travel_direction == "backward":
            wave_direction = 1

        pos_tracker = ValueTracker(0)
        t_tracker = ValueTracker(0)

        distances = [0]  # cumulative distances
        for i in range(1, len(path_points)):
            d = np.linalg.norm(path_points[i] - path_points[i - 1])
            distances.append(distances[-1] + d)
        total_length = distances[-1]

        def wave_pos_along_path(alpha):  # alpha = 0 → start, 1 → end
            target_dist = alpha * total_length
            # find which segment the wave is on
            for i in range(1, len(distances)):
                if target_dist <= distances[i]:
                    segment_alpha = (target_dist - distances[i - 1]) / (
                        distances[i] - distances[i - 1]
                    )
                    return path_points[i - 1] + segment_alpha * (
                        path_points[i] - path_points[i - 1]
                    )
            return path_points[-1]

        def get_wave_angle(alpha):
            # compute target distance along path
            target_dist = alpha * total_length
            # find segment
            for i in range(1, len(distances)):
                if target_dist <= distances[i]:
                    segment_vector = path_points[i] - path_points[i - 1]
                    return np.arctan2(segment_vector[1], segment_vector[0])
            # last segment
            segment_vector = path_points[-1] - path_points[-2]
            return np.arctan2(segment_vector[1], segment_vector[0])

        wave = always_redraw(
            lambda: FunctionGraph(
                lambda x: amplitude
                * np.sin(
                    2
                    * np.pi
                    * frequency
                    * (x + wave_direction * 0.5 * t_tracker.get_value())
                ),
                x_range=[0, 1],
                color=BLACK,
                stroke_width=5,
            )
            .rotate(
                get_wave_angle(pos_tracker.get_value()),
                about_point=wave_pos_along_path(pos_tracker.get_value()),
            )
            .move_to(wave_pos_along_path(pos_tracker.get_value()))
        )

        animations = [
            t_tracker.animate.set_value(3),  # wave oscillation
            pos_tracker.animate.set_value(1),
        ]

        # The trackers are also returned here. This way it's easier
        # to animate complete waves while the multipath bounce
        # is running. For example, we can finish a complete wave
        # while only setting the pos_tracker to 0.6 (= it has
        # traveled 60% of its complete path)
        return wave, animations, t_tracker, pos_tracker


class Moon(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        big_circle = Circle(radius=1, color=BLACK, fill_opacity=1)
        small_circle = Circle(radius=1, color=BLACK,
                              fill_opacity=1).shift(RIGHT * 0.5)

        crescent_moon = Difference(big_circle, small_circle, color=BLACK).set_fill(
            opacity=1, color=BLACK
        )
        self.add(crescent_moon)


class Sun(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # sun center
        sun_center = Circle(radius=0.5, color=BLACK, fill_opacity=1)

        # Rays
        rays = VGroup()
        num_rays = 8
        radius = 0.7
        ray_length = 0.35
        for i in range(num_rays):
            angle = i * (2 * np.pi / num_rays)  # radians

            # Start at the circle boundary
            start = np.array(
                [radius * np.cos(angle), radius * np.sin(angle), 0])
            # End a bit further outward along the same angle
            end = np.array(
                [
                    (radius + ray_length) * np.cos(angle),
                    (radius + ray_length) * np.sin(angle),
                    0,
                ]
            )

            ray = Line(start, end, color=BLACK, stroke_width=8)
            rays.add(ray)

        self.add(rays, sun_center)


class Raindrop(VGroup):
    def __init__(self, stroke_w=6, **kwargs):
        super().__init__(**kwargs)
        # --- base curved line ---
        start = LEFT * 3
        end = RIGHT * 3
        cp1 = start + UP * 4 + LEFT * 0.1
        cp2 = end + DOWN * 0.1 + LEFT * 0.5

        curved_line = CubicBezier(start, cp1, cp2, end, stroke_width=stroke_w)
        curved_line.set_stroke(BLACK, 4).rotate(PI / 2)
        curved_line2 = curved_line.copy().flip(UP)

        self.add(curved_line, curved_line2)


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
        wheel_left.move_to(car_body.get_left() +
                           np.array([0.8, wheel_offset_y, 0]))
        wheel_right.move_to(car_body.get_right() +
                            np.array([-0.8, wheel_offset_y, 0]))

        # group
        self.add(car_body, wheel_left, wheel_right)
        self.move_to(ORIGIN)


class Tree(VGroup):
    def __init__(self, tree_type=1, branch=False, crown_color=None, **kwargs):
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
            color=BLACK,
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
                color=BLACK,
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

            crown = Union(c1, c2, c3, c4, c5).set_stroke(color=BLACK, width=8)

        elif tree_type == 2:
            c1 = Circle(radius=0.5).shift(0.4 * RIGHT + UP * 0.25)
            c2 = Circle(radius=0.6).shift(0.5 * RIGHT + DOWN * 0.3)
            c3 = Circle(radius=0.7).shift(0.5 * LEFT + DOWN * 0.25)
            c4 = Circle(radius=0.5).shift(0 * LEFT + UP * 0.4)

            crown = Union(c1, c2, c3, c4).set_stroke(color=BLACK, width=8)

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


class House(VGroup):
    def __init__(self, wall_height=2, base_width=3, stroke_w=8, **kwargs):
        super().__init__(**kwargs)

        # base of the house
        bottom = Line(
            start=LEFT * (base_width / 2),
            end=RIGHT * (base_width / 2),
            stroke_width=stroke_w,
            color=BLACK,
        ).shift(1 * DOWN)

        left_wall = Line(
            start=bottom.get_start(),
            end=bottom.get_start() + UP * wall_height,
            stroke_width=stroke_w,
            color=BLACK,
        )

        right_wall = Line(
            start=bottom.get_end(),
            end=bottom.get_end() + UP * wall_height,
            stroke_width=stroke_w,
            color=BLACK,
        )

        house_base = VGroup(left_wall, right_wall, bottom)

        # roof
        roof_peak = ORIGIN + UP * 2.05
        roof_left = Line(
            start=LEFT * 1.8 + UP * 0.83,
            end=roof_peak + 0.02 * RIGHT,
            stroke_width=stroke_w,
            color=BLACK,
        )
        roof_right = Line(
            start=RIGHT * 1.8 + UP * 0.83,
            end=roof_peak + 0.02 * LEFT,
            stroke_width=stroke_w,
            color=BLACK,
        )
        roof = VGroup(roof_left, roof_right)

        # chimney
        chimney_width = 0.3
        chimney_height = 0.6
        chimney_bottom_left = roof_left.get_start() + 0.4 * (
            roof_left.get_end() - roof_left.get_start()
        )

        bl = chimney_bottom_left
        tl = chimney_bottom_left + UP * 0.98 * chimney_height
        tr = chimney_bottom_left + UP * 0.98 * chimney_height + RIGHT * chimney_width
        br = chimney_bottom_left + RIGHT * chimney_width + 0.22 * UP

        chimney = Polygon(
            bl,
            tl,
            tr,
            br,
            stroke_width=stroke_w,
            stroke_color=BLACK,
            fill_color=BLACK,
            fill_opacity=1,
        )

        # add to scene (could also group it again)
        self.add(house_base, roof, chimney)


class Crosshair(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        outer_circle = Circle(radius=1.5, color=BLACK, stroke_width=10)
        inner_circle = Circle(radius=1, color=BLACK, stroke_width=7)

        horizontal1 = Line(1.8 * LEFT, 0.6 * LEFT, color=BLACK, stroke_width=8)
        horizontal2 = Line(0.6 * RIGHT, 1.8 * RIGHT,
                           color=BLACK, stroke_width=8)
        vertical1 = Line(0.6 * UP, 1.8 * UP, color=BLACK, stroke_width=8)
        vertical2 = Line(0.6 * DOWN, 1.8 * DOWN, color=BLACK, stroke_width=8)

        plus1 = Line(0.3 * UP, 0.3 * DOWN, color=BLACK, stroke_width=4)
        plus2 = Line(0.3 * LEFT, 0.3 * RIGHT, color=BLACK, stroke_width=4)

        self.add(
            horizontal1,
            horizontal2,
            vertical1,
            vertical2,
            outer_circle,
            inner_circle,
            plus1,
            plus2,
        )


class SecretDocument(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        w, h = 3, 4
        corner = 0.5

        # Document shape with top-right corner cut out
        doc_points = [
            [-w / 2, -h / 2, 0],
            [w / 2, -h / 2, 0],
            [w / 2, h / 2 - corner, 0],
            [w / 2 - corner, h / 2, 0],
            [-w / 2, h / 2, 0],
        ]
        self.paper = Polygon(
            *doc_points,
            color=WHITE,
            fill_color="#f8f8f8",
            fill_opacity=1,
            stroke_width=4,
            stroke_color=BLACK,
            z_index=0
        )

        # Inward fold
        self.fold = Polygon(
            [w / 2 - corner, h / 2, 0],
            [w / 2 - corner, h / 2 - corner, 0],
            [w / 2, h / 2 - corner, 0],
            color=WHITE,
            fill_color="#d0d0d0",
            fill_opacity=1,
            stroke_width=4,
            stroke_color=BLACK,
            z_index=1,
        )

        # Lines of text
        self.lines = VGroup(
            *[
                Line(start=[-1.1, y, 0], end=[1.1, y, 0],
                     stroke_width=3, color=GRAY_C)
                for y in [1, 0.5, 0, -0.5, -1]
            ]
        )

        # SECRET stamp
        self.secret_box = Rectangle(
            width=3.5, height=0.95, color=BLACK, fill_opacity=1, stroke_width=2
        ).move_to([0, -0.9, 0])

        self.secret_text = Text(
            "SECRET", color=WHITE, font_size=42, weight=BOLD
        ).move_to(self.secret_box.get_center())

        self.secret_group = VGroup(self.secret_box, self.secret_text)

        # Add all parts to this VGroup so you can animate them
        self.add(self.paper, self.fold, self.lines, self.secret_group)
