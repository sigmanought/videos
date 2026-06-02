import numpy as np
from manim import *
from scipy.constants import speed_of_light
from utils.colors import *
from utils.decoration import create_shadow
from utils.objects import Car, Clock, Globe
from utils.sat import MeshReflectorSat

fast = 1
COLOR_ADJ = RED
COLOR_OPP = ManimColor("#449c85")
# Replace this with the actual image
FILES_DIR = "./02_bridges_in_stripmap/pngs/"
SAR_IMAGE_PATH = f"{FILES_DIR}/placeholder.png"


class Scene6_p0(MovingCameraScene):
    def construct(self):
        self.camera.background_color = "#e6d8bc"
        sar_image = ImageMobject(SAR_IMAGE_PATH).scale(2)
        scale_width = 1.2
        scale_height = 1.2
        sar_image.stretch_to_fit_width(10 * scale_width)
        sar_image.stretch_to_fit_height(
            2.78 * scale_height
        )  # aspect ratio 3.6 (-> 10/3.6 = 2.78)

        sar_source = (
            Text(
                "Image source: Capella Space, CC BY 4.0 (creativecommons.org/licenses/by/4.0/)",
                font_size=SOURCE_FONT_SIZE,
                font="Zalando Sans",
            )
            .set_color(SOURCE_COLOR)
            .scale(SOURCE_SCALE)
        )
        sar_source.to_corner(DL, buff=0.2)

        grid = Rectangle(
            width=10 * scale_width,
            height=2.78 * scale_height,
            grid_xstep=0.3,
            grid_ystep=0.3,
            stroke_color=WHITE,
            stroke_width=0.03,
            fill_opacity=0,
        )

        sar_shadow = create_shadow(
            sar_image, layers=20, scale_factor=1.1, max_opacity=0.1
        )
        # 00:20
        self.play(FadeIn(sar_shadow, sar_image, sar_source), run_time=fast * 1)

        arrow_origin = sar_image.get_corner(UP + LEFT) + (0.7 * UP + 0.7 * LEFT)
        down_factor = 4.5
        left_factor = 5

        arrow_up = Arrow(
            start=arrow_origin,
            end=arrow_origin + DOWN * down_factor,
            stroke_color=BLACK,
            stroke_width=3,
            tip_length=0.2,
            buff=0,
        )

        arrow_right = Arrow(
            start=arrow_origin,
            end=arrow_origin + RIGHT * left_factor,
            stroke_color=BLACK,
            stroke_width=3,
            tip_length=0.2,
            buff=0,
        )

        self.wait(3)
        self.play(FadeIn(grid))
        self.wait(1)

        # Calculate number of boxes
        num_cols = int(10 * scale_width / 0.3)
        num_rows = int(2.9 * scale_height / 0.3)

        # Create individual filled boxes with gradient
        boxes = []
        total_boxes = num_rows * num_cols

        # Create individual filled boxes
        box_index = 0
        boxes = []
        for col in range(num_cols):
            for row in reversed(range(num_rows)):
                # Calculate progress (0 to 1)
                progress = box_index / (total_boxes - 1)
                box_index += 1

                # Interpolate color from RED to BLUE
                box_color = interpolate_color(RED, BLUE, progress)
                box = Rectangle(
                    width=0.31,
                    height=0.31,
                    stroke_width=0,
                    fill_color=box_color,  # Choose your color
                    fill_opacity=0.5,
                )
                # Position the box
                # Position the box
                x_pos = grid.get_left()[0] - 0.005 + 0.15 + col * 0.3
                y_pos = grid.get_bottom()[1] + 0.15 + row * 0.304
                box.move_to([x_pos, y_pos, 0])
                boxes.append(box)

        text_azimuth = (
            Text(
                "Azimuth", font="Zalando Sans", font_size=SOURCE_FONT_SIZE, color=BLACK
            )
            .move_to(arrow_right.get_end() + 1 * RIGHT)
            .scale(SOURCE_SCALE * 2)
        )
        text_range = (
            Text("Range", font="Zalando Sans", font_size=SOURCE_FONT_SIZE, color=BLACK)
            .move_to(arrow_up.get_end() + 0.4 * DOWN)
            .scale(SOURCE_SCALE * 2)
        )

        arrow_up = Arrow(
            start=arrow_origin,
            end=arrow_origin + DOWN * down_factor,
            stroke_color=BLACK,
            stroke_width=3,
            tip_length=0.2,
            buff=0,
        )

        arrow_up.shift(0.1 * DOWN + 0.4 * RIGHT)
        text_range.shift(0.1 * DOWN + 0.4 * RIGHT)
        arrow_right.shift(0.1 * DOWN + 0.4 * RIGHT)
        text_azimuth.shift(0.1 * DOWN + 0.4 * RIGHT)
        self.wait(3)
        self.play(
            FadeIn(arrow_right, arrow_up, text_azimuth, text_range, text_azimuth),
            run_time=1,
        )
        self.wait(1)
        row_indices = [3, 1, 8, 5]

        for i, row_index in enumerate(row_indices):
            row_copy = [
                box.copy().set_fill(opacity=0.8).set_color(RED).shift(0.01 * UP)
                for box in [
                    boxes[col * num_rows + row_index] for col in range(num_cols)
                ]
            ]
            self.play(FadeIn(*row_copy))
            self.wait(0.5)
            if i < len(row_indices) - 1:
                self.play(FadeOut(*row_copy))

        self.wait(1)
        self.play(FadeOut(*self.mobjects.copy()))


class Scene6_p1(MovingCameraScene):
    """Point spread response:
    slow vs. fast time & along-track vs. range coord systems."""

    def make_arrow_updater(
        self, start, end, grow_duration, travel_duration, speed=2, second_end=None
    ):
        shift_vector = end - start
        direction = normalize(shift_vector)
        arrow_length = 0.7
        distance = np.linalg.norm(shift_vector)
        travel_duration = distance / speed
        cycle = 2 * travel_duration
        elapsed_time = 0

        def arrow_updater(mob, dt):
            nonlocal elapsed_time, end, shift_vector, direction, start
            elapsed_time += dt
            t = elapsed_time

            if t < grow_duration:
                return Arrow(
                    start, start + direction * arrow_length, buff=0, color=BLACK
                )
            else:
                travel_t = t - grow_duration
                if travel_t >= cycle:
                    mob.set_opacity(0)
                    mob.clear_updaters()
                    return
                progress = travel_t / travel_duration
                if progress <= 1:
                    alpha = progress
                    pos = interpolate(start, end - direction * arrow_length, alpha)
                    tip = pos + direction * arrow_length
                    mob.become(Arrow(start=pos, end=tip, buff=0, color=BLACK))
                else:
                    if second_end is not None:
                        start = second_end
                        shift_vector = end - start
                        direction = normalize(shift_vector)
                    alpha = 2 - progress
                    pos = interpolate(start, end - direction * arrow_length, alpha)
                    tip = pos + direction * arrow_length
                    mob.become(Arrow(start=tip, end=pos, buff=0, color=BLACK))

        return arrow_updater

    @staticmethod
    def draw_psr(psr_loc, axes, axes2, color, sat):
        """Draw a hyperbolic point spread response from a point scatterer
        located at psr_loc where sat is the moving satellite."""
        u = ValueTracker(0)
        x, r = psr_loc
        graph = always_redraw(
            lambda: axes.plot_parametric_curve(
                lambda t: np.array([t, -np.sqrt((t - x) ** 2 + r**2), 0]),
                t_range=[0, u.get_value()],
                color=color,
            )
        )
        point_scatterer = Dot(axes2.c2p(psr_loc[0], -psr_loc[1]), color=color)
        point_scatterer.z_index = 5
        connection_line = always_redraw(
            lambda: Line(sat.get_center(), point_scatterer.get_center(), color=GREY_C)
        )
        return u, graph, point_scatterer, connection_line

    def construct(self):

        fw = config.frame_width
        fh = config.frame_height

        axes = Axes(
            x_range=[-fw / 2, fw / 2, 1],
            y_range=[-fh / 2, fh / 2, 1],
            x_length=fw,
            y_length=fh,
            tips=False,
        ).set_color(GREY)
        axes.add_coordinates(color=GREY)
        self.play_scene_6()

    def play_scene_6(self):
        self.camera.background_color = "#e6d8bc"
        radius = 1
        clock = Clock(radius=radius).shift(3 * LEFT)
        globe = Globe(radius=radius).shift(3 * RIGHT)

        self.play((FadeIn(clock, globe)), run_time=1)
        self.wait(1)
        self.play(
            Succession(Indicate(clock, color=BLACK), Indicate(globe, color=BLACK)),
            run_time=2,
        )
        self.play(FadeOut(clock), run_time=1)

        axes = (
            Axes(
                x_range=[0, 4, 1],
                y_range=[-8, 0],
                x_length=5,
                y_length=3,
                axis_config={"include_numbers": True, "tip_length": 0.05},
                x_axis_config={
                    "include_tip": False,
                    "numbers_to_include": [0, 1, 2, 3],
                    "label_direction": UP,
                    "tick_size": 0,
                },
                y_axis_config={
                    "include_tip": False,
                    "numbers_to_include": [-6, -4, -2],
                    "label_direction": LEFT,
                    "tick_size": 0,
                },
            )
            .set_color(BLACK)
            .shift(LEFT * 3.25 + DOWN * 1)
        )
        axes.get_y_axis().numbers.set_opacity(0)
        axes.get_x_axis().numbers.set_opacity(0)

        axes.y_axis.numbers.set_direction(LEFT)
        axes.y_axis.numbers  # .shift(LEFT * 0.3)

        x_numbers = axes.get_x_axis().numbers
        new_labels = [
            "\\textsf{1}~\\textsf{sec}",
            "\\textsf{2}~\\textsf{sec}",
            "\\textsf{3}~\\textsf{sec}",
        ]
        y_numbers = axes.get_y_axis().numbers

        r_0 = 3
        u_0 = 1
        x_point = np.sqrt((u_0 - 0) ** 2 + r_0**2)
        rs2 = [644, 645, 646]
        ts = [(r * 1000) * 2 / speed_of_light * (1e6) for r in rs2]
        ts.reverse()
        new_y_labels = [rf"\textsf{{{t/1000:.3f}}}" + "~\\textsf{ms}" for t in ts]

        for number, new_text in zip(y_numbers, new_y_labels):
            new_label = MathTex(
                new_text, color=BLACK, font_size=SOURCE_FONT_SIZE
            ).scale(SOURCE_SCALE * 1.5)
            new_label.move_to(number.get_center() + 0.2 * LEFT)
            number.become(new_label)
        axes.get_y_axis().numbers.set_opacity(0)

        for number, new_text in zip(x_numbers, new_labels):
            new_label = MathTex(
                new_text, color=BLACK, font_size=SOURCE_FONT_SIZE
            ).scale(SOURCE_SCALE * 1.5)
            new_label.move_to(number.get_center())
            number.become(new_label)
        axes.get_x_axis().numbers.set_opacity(0)

        left_tip = Arrow(
            start=axes.x_axis.get_start(),
            end=axes.x_axis.get_end() + 0.1 * RIGHT,
            buff=0,
            stroke_width=2,
            color=BLACK,
            tip_length=0.15,
        )
        axes.add(left_tip)

        right_tip = Arrow(
            start=axes.y_axis.get_end(),
            end=axes.y_axis.get_start() + 0.1 * DOWN,
            buff=0,
            stroke_width=2,
            color=BLACK,
            tip_length=0.15,
        )
        axes.add(right_tip)

        tick_positions = [0, 1, 2, 3]
        tick_length = 0.1
        ticks_x = VGroup()
        for x in tick_positions:
            tick = Line(
                start=axes.coords_to_point(x, 0) + UP * tick_length / 2,
                end=axes.coords_to_point(x, 0) + DOWN * tick_length / 2,
                stroke_width=2,
                color=BLACK,
            )
            ticks_x.add(tick)

        tick_positions = [-6, -4, -2]
        tick_length = 0.1
        ticks_y = VGroup()
        for x in tick_positions:
            tick = Line(
                start=axes.coords_to_point(0, x) + LEFT * tick_length / 2,
                end=axes.coords_to_point(0, x) + RIGHT * tick_length / 2,
                stroke_width=2,
                color=BLACK,
            )
            ticks_y.add(tick)

        x_label = (
            axes.get_x_axis_label("\mathsf{u}")
            .next_to(axes.x_axis.get_end(), RIGHT)
            .set_color(BLACK)
        )

        labels = axes.get_axis_labels(
            x_label="",
            y_label=MathTex(r"\mathsf{t}"),
        ).set_color(BLACK)
        y_label = labels[1]
        y_label.next_to(axes.y_axis.get_start(), DOWN)
        y_label_prop = (
            MathTex(r"= \frac{\mathsf{2}}{\mathsf{c}}\mathsf{r}", color=BLACK)
            .next_to(y_label, RIGHT, buff=-0.1)
            .set_stroke(BLACK)
            .scale(0.6)
        )

        new_y_label = (
            Text(
                "fast time",
                font="Zalando Sans",
                color=BLACK,
                font_size=SOURCE_FONT_SIZE,
            )
            .scale(SOURCE_SCALE * 1.5)
            .set_color(BLACK)
        )
        new_y_label.move_to(y_label)

        new_x_label = (
            Text(
                "slow\ntime",
                font="Zalando Sans",
                color=BLACK,
                font_size=SOURCE_FONT_SIZE,
            )
            .scale(SOURCE_SCALE * 1.5)
            .set_color(BLACK)
        )
        new_x_label.move_to(x_label).shift(0.2 * RIGHT)

        shift_title = 0.8
        title = (
            VGroup(
                Text(
                    "Raw Data Space",
                    font="Zalando Sans",
                    color=BLACK,
                    font_size=SOURCE_FONT_SIZE,
                )
                .scale(SOURCE_SCALE * 2)
                .set_stroke(BLACK),
                Clock(radius=0.2, stroke_width=2.5),
            )
            .arrange(RIGHT, buff=0.15)
            .next_to(axes, UP)
            .shift(0.4 * RIGHT + shift_title * UP)
        )

        sat = (
            MeshReflectorSat(radius=2, num_scallops=12, scallop_depth=0.7)
            .set_stroke(width=3)
            .scale(0.1)
            .rotate(PI)
        )
        sat.z_index = 10
        u = ValueTracker(0)
        sat.add_updater(lambda m: m.move_to(axes.c2p(u.get_value(), 0)))

        self.play(Create(axes), run_time=fast * 1)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.move_to(axes.c2p(2, -3.5)).scale(0.65))
        self.play(Create(ticks_x), run_time=fast * 1)

        self.play(axes.get_x_axis().numbers.animate.set_opacity(1), run_time=fast * 1)
        self.play(Write(x_label), run_time=fast * 1)

        self.play(ReplacementTransform(x_label, new_x_label), run_time=fast * 1)
        old_x_label = (
            MathTex("\mathsf{u}")
            .set_color(BLACK)
            .move_to(axes.x_axis.get_end())
            .next_to(axes.x_axis.get_end(), RIGHT)
        )

        x_numbers = axes.get_x_axis().numbers
        new_labels_km = [
            "\\textsf{5}~\\textsf{km}",
            "\\textsf{10}~\\textsf{km}",
            "\\textsf{15}~\\textsf{km}",
        ]

        arrows = VGroup()
        km_labels = VGroup()

        for number, text in zip(x_numbers, new_labels_km):
            km_label = MathTex(text, color=BLACK, font_size=SOURCE_FONT_SIZE).scale(
                SOURCE_SCALE * 1.5
            )
            km_label.next_to(number, UP, buff=0.4)
            arrow = Arrow(
                start=number.get_top(),
                end=km_label.get_bottom(),
                buff=0.05,
                stroke_width=0.75,
                color=BLACK,
            )
            km_labels.add(km_label)
            arrows.add(arrow)

        self.play(self.camera.frame.animate.shift(0.5 * UP))
        self.play(FadeIn(arrows), run_time=fast * 1)
        self.play(FadeIn(km_labels), run_time=fast * 1)
        self.wait(fast * 2)
        self.play(FadeOut(arrows, km_labels), run_time=fast * 1)
        self.play(self.camera.frame.animate.shift(0.5 * DOWN))

        self.play(FadeIn(sat), run_time=fast * 1)
        self.wait(1)
        self.play(u.animate.set_value(4), run_time=fast * 4, rate_func=linear)
        sat.clear_updaters()
        sat.move_to(axes.c2p(0, 0))
        self.play(ReplacementTransform(new_x_label, old_x_label), run_time=fast * 1)

        self.play(Create(ticks_y), run_time=fast * 1)
        self.play(axes.get_y_axis().numbers.animate.set_opacity(1), run_time=1)
        self.play(FadeIn(labels), run_time=fast * 1)

        self.play(ReplacementTransform(y_label, new_y_label), run_time=fast * 1)
        self.wait(fast * 0.5)
        old_y_label = MathTex("\mathsf{t}").set_color(BLACK).move_to(new_y_label)
        self.play(ReplacementTransform(new_y_label, old_y_label), run_time=fast * 1)
        self.play(Write(y_label_prop), run_time=fast * 1)
        self.wait(fast * 0.5)
        self.play(FadeOut(y_label_prop), run_time=fast * 1)

        self.wait(4)

        sat2 = (
            MeshReflectorSat(radius=2, num_scallops=12, scallop_depth=0.7)
            .set_stroke(width=3)
            .scale(0.1)
            .rotate(PI)
        )
        sat2.set_z_index(10, family=True)

        axes2 = (
            Axes(
                x_range=[0, 4, 1],
                y_range=[-4, 0, 1],
                x_length=5,
                y_length=3,
                x_axis_config={
                    "include_tip": False,
                    "numbers_to_include": [0, 1, 2, 3],
                    "label_direction": UP,
                    "tick_size": 0,
                },
                y_axis_config={
                    "include_tip": False,
                    "numbers_to_include": [-3, -2, -1, 0],
                    "label_direction": LEFT,
                    "tick_size": 0,
                },
            )
            .set_color(BLACK)
            .shift(RIGHT * 3.75 + DOWN * 1)
        )
        axes2.y_axis.numbers.set_direction(LEFT)

        tick_positions = [-3, -2, -1, 0]
        tick_length = 0.1
        ticks_y2 = VGroup()
        for x in tick_positions:
            tick = Line(
                start=axes2.coords_to_point(0, x) + LEFT * tick_length / 2,
                end=axes2.coords_to_point(0, x) + RIGHT * tick_length / 2,
                stroke_width=2,
                color=BLACK,
            )
            ticks_y2.add(tick)

        left_tip2 = Arrow(
            start=axes2.x_axis.get_start(),
            end=axes2.x_axis.get_end() + 0.1 * RIGHT,
            buff=0,
            stroke_width=2,
            color=BLACK,
            tip_length=0.15,
        )
        axes2.add(left_tip2)

        right_tip2 = Arrow(
            start=axes2.y_axis.get_end(),
            end=axes2.y_axis.get_start() + 0.1 * DOWN,
            buff=0,
            stroke_width=2,
            color=BLACK,
            tip_length=0.15,
        )
        axes2.add(right_tip2)

        x_numbers2 = axes2.get_x_axis().numbers
        new_labels2 = [
            "\\textsf{5}~\\textsf{km}",
            "\\textsf{10}~\\textsf{km}",
            "\\textsf{15}~\\textsf{km}",
        ]

        y_numbers2 = axes2.get_y_axis().numbers
        new_y_labels2 = [
            "\\textsf{645}~\\textsf{km}",
            "\\textsf{644}~\\textsf{km}",
            "\\textsf{643}~\\textsf{km}",
        ]

        for number, new_text in zip(y_numbers2, new_y_labels2):
            new_label = MathTex(
                new_text, color=BLACK, font_size=SOURCE_FONT_SIZE
            ).scale(SOURCE_SCALE * 1.5)
            new_label.move_to(number.get_center() + 0.2 * LEFT)
            number.become(new_label)
        axes2.get_y_axis().numbers.set_opacity(0)

        for number, new_text in zip(x_numbers2, new_labels2):
            new_label = MathTex(
                new_text, color=BLACK, font_size=SOURCE_FONT_SIZE
            ).scale(SOURCE_SCALE * 1.5)
            new_label.move_to(number.get_center())
            number.become(new_label)
        axes2.get_x_axis().numbers.set_opacity(0)

        tick_positions = [0, 1, 2, 3]
        tick_length = 0.1
        ticks2 = VGroup()
        for x in tick_positions:
            tick = Line(
                start=axes2.coords_to_point(x, 0) + UP * tick_length / 2,
                end=axes2.coords_to_point(x, 0) + DOWN * tick_length / 2,
                stroke_width=2,
                color=BLACK,
            )
            ticks2.add(tick)

        x_label2 = (
            axes2.get_x_axis_label("\mathsf{x}")
            .next_to(axes2.x_axis.get_end(), RIGHT)
            .set_color(BLACK)
        )
        labels2 = axes2.get_axis_labels(
            x_label=MathTex(""),
            y_label=MathTex(r"\mathsf{r}"),
        ).set_color(BLACK)
        y_label2 = labels2[1]
        y_label2.next_to(axes2.y_axis.get_start(), DOWN)
        y_label_prop = (
            MathTex(r"= \frac{\mathsf{2}}{\mathsf{c}}\mathsf{r}", color=BLACK)
            .next_to(y_label2, RIGHT, buff=-0.1)
            .set_stroke(BLACK)
            .scale(0.5)
        )

        psr_loc = [1, 3]
        scatterer_location = Dot(axes2.c2p(psr_loc[0], -psr_loc[1]), color=RED_E)
        scatterer_location.z_index = 3
        text_scat_loc = (
            Text("point scatterer", font="Zalando Sans", font_size=SOURCE_FONT_SIZE)
            .scale(SOURCE_SCALE * 1.5)
            .set_stroke(BLACK)
        )
        text_scat_loc.next_to(scatterer_location, direction=RIGHT).set_color(BLACK)

        title2 = VGroup(
            Text(
                "Acquisition Geometry",
                font="Zalando Sans",
                color=BLACK,
                font_size=SOURCE_FONT_SIZE,
            )
            .scale(SOURCE_SCALE * 2)
            .set_stroke(BLACK),
            Globe(radius=0.2, stroke_width=2.5),
        ).arrange(RIGHT, buff=0.15)
        title2.next_to(axes2, UP).shift(0.8 * RIGHT + shift_title * UP)

        new_y_label2 = (
            Paragraph(
                "across-",
                "track",
                font="Zalando Sans",
                color=BLACK,
                font_size=SOURCE_FONT_SIZE,
                alignment="center",
            )
            .scale(SOURCE_SCALE * 1.5)
            .set_color(BLACK)
        )
        new_y_label2.move_to(y_label2).shift(0.1 * DOWN)

        new_x_label2 = (
            Paragraph(
                "along-",
                "track",
                font="Zalando Sans",
                color=BLACK,
                font_size=SOURCE_FONT_SIZE,
                alignment="center",
            )
            .scale(SOURCE_SCALE * 1.5)
            .set_color(BLACK)
        )
        new_x_label2.move_to(x_label2).shift(0.3 * RIGHT + 0.05 * DOWN)

        self.play(Restore(self.camera.frame))

        self.play(Indicate(globe, color=BLACK), run_time=1)
        self.play(FadeOut(globe), run_time=1)
        self.play(Create(axes2), run_time=fast * 1)
        self.play(
            self.camera.frame.animate.move_to(axes2.c2p(2, -2)).scale(0.65),
            axes.animate.set_opacity(0),
            sat.animate.set_opacity(0),
            ticks_y.animate.set_opacity(0),
            labels.animate.set_opacity(0),
            ticks_x.animate.set_opacity(0),
            old_x_label.animate.set_opacity(0),
        )

        self.play(Create(ticks2), run_time=fast * 1)
        self.play(axes2.get_x_axis().numbers.animate.set_opacity(1), run_time=fast * 1)
        self.wait(fast * 1)
        self.wait(4)

        old_x_label2 = (
            MathTex(r"\mathsf{x}")
            .set_color(BLACK)
            .next_to(axes2.x_axis.get_end(), RIGHT)
        )
        old_y_label2 = (
            MathTex(r"\mathsf{r}")
            .set_color(BLACK)
            .next_to(axes2.y_axis.get_start(), DOWN)
        )

        self.play(Write(x_label2), run_time=fast * 1)
        self.play(Transform(x_label2, new_x_label2), run_time=fast * 1)
        self.wait(1)
        self.play(Transform(x_label2, old_x_label2), run_time=fast * 1)

        ### 3 SEC, total 47 SEC ###
        self.play(FadeIn(ticks_y2), run_time=fast * 1)
        self.play(axes2.get_y_axis().numbers.animate.set_opacity(1))
        self.play(FadeIn(labels2), run_time=fast * 1)

        self.wait(0.5)
        self.play(Transform(y_label2, new_y_label2), run_time=fast * 1)
        self.wait(0.5)
        self.play(Transform(y_label2, old_y_label2), run_time=fast * 1)

        ### 6 SEC, total 54 SEC ###
        self.wait(3)
        self.play(
            Restore(self.camera.frame),
            axes.animate.set_opacity(1),
            sat.animate.set_opacity(1),
            ticks_y.animate.set_opacity(1),
            labels.animate.set_opacity(1),
            ticks_x.animate.set_opacity(1),
            old_x_label.animate.set_opacity(1),
        )
        self.play(
            FadeIn(title),
            self.camera.frame.animate.shift(DOWN * 0.25).scale(1.02),
            run_time=fast * 1,
        )
        self.play(Indicate(title), color=BLACK, run_time=1)

        ### 10 SEC, total 1:04 ###

        self.play(FadeIn(title2), run_time=fast * 1)
        self.play(Indicate(title2), color=BLACK, run_time=fast * 1)

        new_y_label2 = (
            Paragraph(
                "lat",
                font="Zalando Sans",
                color=BLACK,
                font_size=SOURCE_FONT_SIZE,
                alignment="center",
            )
            .scale(SOURCE_SCALE * 1.5)
            .set_color(BLACK)
        )
        new_y_label2.move_to(y_label2).shift(0.1 * UP)

        new_x_label2 = (
            Paragraph(
                "lon",
                font="Zalando Sans",
                color=BLACK,
                font_size=SOURCE_FONT_SIZE,
                alignment="center",
            )
            .scale(SOURCE_SCALE * 1.5)
            .set_color(BLACK)
        )
        new_x_label2.move_to(x_label2).shift(0.1 * LEFT)

        self.play(
            AnimationGroup(
                Transform(y_label2, new_y_label2), Transform(x_label2, new_x_label2)
            ),
            run_time=fast * 1,
        )
        old_y_label2 = (
            MathTex(r"\textsf{r}")
            .set_color(BLACK)
            .move_to(new_y_label2)
            .shift(0.1 * DOWN)
        )
        old_x_label2 = (
            MathTex(r"\textsf{x}")
            .set_color(BLACK)
            .move_to(new_x_label2)
            .shift(0.1 * RIGHT)
        )
        self.play(
            AnimationGroup(
                Transform(y_label2, old_y_label2),
                Transform(x_label2, old_x_label2),
                run_time=fast * 1,
            )
        )
        self.wait(fast * 1)

        self.play(Indicate(title), color=BLACK, run_time=fast * 1)

        y_level = title.get_center()[1]

        arrow = Arrow(
            start=np.array([title.get_right()[0], y_level, 0]) + 0.5 * RIGHT,
            end=np.array([title2.get_left()[0], y_level, 0]) + 0.5 * LEFT,
            color=BLACK,
            stroke_width=1.5,
            tip_length=0.1,
        )

        self.play(Create(arrow), run_time=fast * 1)
        self.play(Indicate(title2), color=BLACK, run_time=fast * 1)
        self.wait(fast * 1)
        self.play(FadeOut(arrow), run_time=fast * 1)

        ### 10 SEC, total 1:14 ###

        text_scatterer = MathTex(
            r"(\textsf{x}^*, \textsf{r}^*)", font_size=SOURCE_FONT_SIZE
        ).scale(SOURCE_SCALE * 2.5)
        text_scatterer.next_to(scatterer_location, direction=DOWN).set_color(
            BLACK
        ).set_stroke(BLACK)

        self.wait(fast * 1)
        self.play(FadeIn(scatterer_location), run_time=fast * 1)
        self.play(Write(text_scatterer), run_time=fast * 1)
        self.play(
            AnimationGroup(
                Indicate(scatterer_location, color=scatterer_location.get_color()),
                Indicate(text_scatterer, color=BLACK),
            ),
            run_time=fast * 1,
        )

        self.wait(1)
        question_mark = (
            Text("?", color=scatterer_location.get_color())
            .set_stroke(scatterer_location.get_color())
            .move_to(axes.c2p(2, -4.5))
            .scale(3)
        )
        self.play(Write(question_mark), run_time=fast * 1)
        self.play(
            Indicate(question_mark, color=scatterer_location.get_color()),
            run_time=fast * 1,
        )

        ### 7 SEC, total 1:21 ###

        scatterer_location_old = scatterer_location.copy()
        self.wait(fast * 1)
        self.play(FadeIn(text_scat_loc), run_time=fast * 1)

        car = (
            Car()
            .scale(0.15)
            .move_to(scatterer_location)
            .set_color(RED_E)
            .flip(UP)
            .set_stroke(width=2)
            .set_fill(RED_E, opacity=0.2)
        )
        self.play(Transform(scatterer_location, car), run_time=fast * 1)
        self.wait(fast * 1)
        self.play(
            Transform(scatterer_location, scatterer_location_old), run_time=fast * 1
        )

        u, graph, _, connection_line = self.draw_psr(
            psr_loc=psr_loc, axes=axes, axes2=axes2, color=RED_E, sat=sat2
        )
        self.add(graph)

        ### 5 SEC, total 1:26 ###

        sat.add_updater(lambda m: m.move_to(axes.c2p(u.get_value(), 0)))
        sat2.add_updater(lambda m: m.move_to(axes2.c2p(u.get_value(), 0)))
        self.play(FadeIn(sat2), run_time=fast * 1)
        self.play(
            AnimationGroup(sat.animate.scale(1.1), sat2.animate.scale(1.1)),
            run_time=fast,
        )
        self.play(
            AnimationGroup(sat.animate.scale(1 / 1.1), sat2.animate.scale(1 / 1.1)),
            run_time=fast,
        )

        self.wait(3)
        sat_coord = sat2.get_center()
        scat_coord = scatterer_location.get_center()
        arrow = Arrow(
            sat_coord,
            sat_coord + normalize(scat_coord - sat_coord) * 0.2,
            buff=0,
            color=BLACK,
        )
        speed = 2
        grow_duration = 0.05
        travel_duration = 1
        arrow_updater_sb = self.make_arrow_updater(
            sat_coord, scat_coord, grow_duration, travel_duration, speed
        )
        arrow.add_updater(arrow_updater_sb)

        self.add(arrow)
        self.wait(2)

        ### 5 SEC, total 1:34 ###

        line_example = Line(
            sat2.get_center(), scatterer_location.get_center(), color=GREY_C
        )

        dot_legs = Dot(axes2.c2p(0, -psr_loc[1]), color=RED_E)
        line_example_leg1 = Line(
            sat2.get_center(), dot_legs.get_center(), color=COLOR_OPP
        )

        line_example_leg2 = Line(
            scatterer_location.get_center(), dot_legs.get_center(), color=COLOR_ADJ
        )
        line_example_leg1.z_index = 2
        line_example_leg2.z_index = 2
        line_example_leg1_text = MathTex(
            r"\textsf{r}^*", color=COLOR_OPP, font_size=SOURCE_FONT_SIZE
        ).scale(SOURCE_SCALE * 2)
        line_example_leg1_text.next_to(line_example_leg1, RIGHT).shift(0.2 * DOWN)
        line_example_leg2_text = (
            MathTex(
                r"\textsf{x}-\textsf{x}^*", color=COLOR_ADJ, font_size=SOURCE_FONT_SIZE
            )
            .next_to(line_example_leg2, DOWN)
            .scale(SOURCE_SCALE * 2)
            .shift(0.1 * UP)
        )

        right_angle = RightAngle(
            line_example_leg1,
            line_example_leg2,
            length=0.2,
            color=GREY_C,
            quadrant=(-1, -1),
        )

        triangle = Polygon(
            sat2.get_center(),
            scatterer_location.get_center(),
            dot_legs.get_center(),
            color=GREY,
            fill_color=GREY_B,
            fill_opacity=0.5,
        )

        scale_factor = 0.8
        travel_path_length = (
            MathTex(
                r"\textsf{t}(",
                r"\textsf{u};" r"\textsf{x}^*",
                r",",
                r"\textsf{r}^*\vphantom{a}",
                r")",
                r"=",
                r"\sqrt{(",
                r"\textsf{x}(\textsf{u})",
                r"-",
                r"\textsf{x}^*)",
                r"^\textsf{2}",
                r"+",
                r"(",
                r"\textsf{r}^*\vphantom{b}",
                r")",
                r"^\textsf{2}}",
                color=BLACK,
            )
            .shift(3 * UP)
            .scale(scale_factor)
            .set_stroke(BLACK)
        )
        travel_path_length[3].set_color(COLOR_OPP)
        travel_path_length[13].set_color(COLOR_OPP)
        travel_path_length[7].set_color(COLOR_ADJ)
        travel_path_length[8].set_color(COLOR_ADJ)
        travel_path_length[9].set_color(COLOR_ADJ)

        travel_path_length_general = (
            MathTex(
                r"\textsf{t}(",
                r"\textsf{u};" r"\textsf{x}^*",
                r",",
                r"\textsf{r}^*\vphantom{a}",
                r")",
                r"=",
                r"\sqrt{(",
                r"\textsf{x}(\textsf{u})",
                r"-",
                r"\textsf{x}^*",
                r")^\textsf{2}",
                r"+",
                r"(",
                r"\textsf{r}^*\vphantom{b}",
                r")^\textsf{2}} \cdot \frac{\textsf{2}}{\textsf{c}}",
                color=BLACK,
            )
            .shift(3 * UP)
            .scale(scale_factor)
            .set_stroke(BLACK)
        )

        travel_path_length_star = (
            MathTex(
                r"\textsf{t}(",
                r"\textsf{0};",
                r"\textsf{x}^*",
                r",",
                r"\textsf{r}^*\vphantom{a}",
                r")",
                r"=",
                r"\sqrt{(",
                r"\textsf{0}",
                r"-",
                r"\textsf{x}^*)",
                r"^\textsf{2}",
                r"+",
                r"(",
                r"\textsf{r}^*)\vphantom{b}",
                r"^\textsf{2}}",
                r"\cdot",
                r"\frac{\textsf{2}}{\textsf{c}}",
                color=BLACK,
            )
            .shift(3 * UP)
            .scale(scale_factor)
            .set_stroke(BLACK)
        )

        travel_path_with_factor1 = (
            MathTex(
                r"\textsf{t}(",
                r"\textsf{u};" r"\textsf{x}^*",
                r",",
                r"\textsf{r}^*\vphantom{a}",
                r")",
                r"=",
                r"\sqrt{(",
                r"\textsf{x}(\textsf{u})",
                r"-",
                r"\textsf{x}^*)",
                r"^\textsf{2}",
                r"+",
                r"(",
                r"\textsf{r}^*)\vphantom{b}",
                r"^\textsf{2}}",
                r"\cdot",
                r"\textsf{2}",
                color=BLACK,
            )
            .move_to(travel_path_length)
            .scale(scale_factor)
        )

        travel_path_with_factor2 = (
            MathTex(
                r"\textsf{t}(",
                r"\textsf{u};",
                r"\textsf{x}^*",
                r",",
                r"\textsf{r}^*\vphantom{a}",
                r")",
                r"=",
                r"\sqrt{(",
                r"\textsf{x}(\textsf{u})",
                r"-",
                r"\textsf{x}^*)",
                r"^\textsf{2}",
                r"+",
                r"(",
                r"\textsf{r}^*)\vphantom{b}",
                r"^\textsf{2}}",
                r"\cdot",
                r"\frac{\textsf{2}}{\textsf{c}}",
                color=BLACK,
            )
            .move_to(travel_path_length)
            .scale(scale_factor)
        )

        travel_path_with_comp1 = (
            MathTex(
                r"\textsf{t}(",
                r"\textsf{0};",
                r"\textsf{x}^*",
                r",",
                r"\textsf{r}^*\vphantom{a}",
                r")",
                r"=",
                r"\sqrt{(",
                r"\textsf{0}",
                r"-",
                r"\textsf{x}^*)",
                r"^\textsf{2}",
                r"+",
                r"(",
                r"\textsf{r}^*)\vphantom{b}",
                r"^\textsf{2}}",
                r"\cdot",
                r"\frac{\textsf{2}}{\textsf{c}}",
                color=BLACK,
            )
            .move_to(travel_path_length)
            .scale(scale_factor)
        )

        travel_path_with_comp2 = (
            MathTex(
                r"\textsf{t}(",
                r"0;",
                r"\textsf{x}^*",
                r",",
                r"\textsf{r}^*\vphantom{a}",
                r")",
                r"\approx",
                r"\textsf{645.02}~\textsf{km}",
                r"",
                r"",
                r"",
                r"",
                r"",
                r"",
                r"\cdot",
                r"\frac{\textsf{2}}{\textsf{c}}",
                color=BLACK,
            )
            .move_to(travel_path_length)
            .scale(scale_factor)
        )

        travel_path_with_comp3 = (
            MathTex(
                r"\textsf{t}(",
                r"0;",
                r"\textsf{x}^*",
                r",",
                r"\textsf{r}^*\vphantom{a}",
                r")",
                r"\approx",
                r"\textsf{1290.04}~\textsf{km}",
                r"\cdot\frac{\textsf{1}}{\textsf{3}} \cdot \textsf{10}^{-\textsf{6}}\textsf{s}",
                r"",
                r"",
                r"",
                r"",
                r"",
                r"",
                r"",
                color=BLACK,
            )
            .move_to(travel_path_length)
            .scale(scale_factor)
        )

        travel_path_with_comp4 = (
            MathTex(
                r"\textsf{t}(",
                r"0;",
                r"\textsf{x}^*",
                r",",
                r"\textsf{r}^*\vphantom{a}",
                r")",
                r"\approx",
                r"\textsf{430.02}\,\mu\,\textsf{s}",
                r"",
                r"",
                r"",
                r"",
                r"",
                r"",
                r"",
                r"",
                color=BLACK,
            )
            .move_to(travel_path_length)
            .scale(scale_factor)
        )

        box = RoundedRectangle(
            width=7,
            height=1.5,
            corner_radius=0.2,
            color=BLACK,
            stroke_width=2,
            fill_opacity=0.65,
            fill_color=LIGHT_BEIGE,
        ).move_to(travel_path_length.get_center())

        cross_size = 0.1
        r_0 = 3
        u_0 = 1
        x_point = np.sqrt((u_0 - 0) ** 2 + r_0**2)
        point = axes.c2p(0, -x_point)
        cross = VGroup(
            Line(
                start=point + LEFT * cross_size + UP * cross_size,
                end=point + RIGHT * cross_size + DOWN * cross_size,
                color=RED_E,
            ),
            Line(
                start=point + LEFT * cross_size + DOWN * cross_size,
                end=point + RIGHT * cross_size + UP * cross_size,
                color=RED_E,
            ),
        )

        self.wait(fast * 1)
        self.play(Create(line_example), run_time=fast * 1)
        self.wait(fast * 1)

        ### 3 SEC, total: 1:37 ###

        self.play(
            Create(line_example_leg1),
            Create(line_example_leg2),
            FadeIn(triangle),
            run_time=fast * 1,
        )
        self.bring_to_front(sat2)
        self.play(Create(right_angle), FadeOut(text_scat_loc), run_time=fast * 1)
        self.play(
            Succession(
                text_scatterer.animate.next_to(scatterer_location, RIGHT),
                FadeIn(line_example_leg1_text),
                FadeIn(line_example_leg2_text),
            ),
            run_time=fast * 1,
        )
        self.play(
            title.animate.shift(0.55 * DOWN),
            title2.animate.shift(0.55 * DOWN),
            run_time=fast * 1,
        )

        self.play(
            FadeIn(box),
            self.camera.frame.animate.shift(UP * 0.65).scale(1.0294118),
            run_time=fast * 1,
        )
        self.play(Succession(Write(travel_path_length)), run_time=fast * 1)
        self.wait(2)

        self.wait(3)
        self.play(
            TransformMatchingShapes(travel_path_length, travel_path_with_factor1),
            run_time=fast * 1,
        )
        travel_path_length = travel_path_with_factor1

        self.wait(fast * 5)
        self.wait(fast * 3)

        self.play(
            TransformMatchingTex(travel_path_length, travel_path_with_factor2),
            run_time=fast * 1,
        )
        travel_path_length = travel_path_with_factor2
        self.wait(1)

        self.wait(0.5)
        self.play(
            TransformMatchingTex(travel_path_length, travel_path_length_star),
            run_time=fast * 0.5,
        )
        travel_path_length = travel_path_length_star

        self.wait(0.5)
        self.play(
            TransformMatchingTex(travel_path_length, travel_path_with_comp1),
            run_time=fast * 0.5,
        )
        travel_path_length = travel_path_with_comp1

        self.wait(0.5)
        self.play(
            TransformMatchingTex(
                travel_path_length,
                travel_path_with_comp2,
                path_arc=0,
                transform_mismatches=False,
            ),
            run_time=fast * 0.5,
        )
        travel_path_length = travel_path_with_comp2

        self.wait(0.5)
        self.play(
            TransformMatchingTex(
                travel_path_length,
                travel_path_with_comp3,
                path_arc=0,
                transform_mismatches=False,
            ),
            run_time=fast * 0.5,
        )
        travel_path_length = travel_path_with_comp3

        self.wait(0.5)
        self.play(
            AnimationGroup(
                TransformMatchingTex(
                    travel_path_length,
                    travel_path_with_comp4,
                    path_arc=0,
                    transform_mismatches=False,
                ),
                FadeOut(question_mark),
                Create(cross),
            ),
            run_time=fast * 1,
        )
        travel_path_length = travel_path_with_comp4
        self.wait(2)

        self.play(
            TransformMatchingTex(
                travel_path_length,
                travel_path_length_general,
                path_arc=0,
                transform_mismatches=False,
            )
        )
        travel_path_length = travel_path_length_general

        self.play(
            FadeOut(
                line_example_leg1,
                line_example_leg2,
                line_example_leg1,
                line_example_leg2,
                line_example,
                line_example_leg1_text,
                line_example_leg2_text,
                right_angle,
                triangle,
            ),
            run_time=fast * 1,
        )
        self.add(connection_line)

        self.play(u.animate.set_value(4), run_time=fast * 5, rate_func=linear)
        self.wait(fast * 1)
        self.play(FadeOut(connection_line), run_time=fast * 1)
        self.remove(connection_line)

        self.play(FadeOut(text_scatterer), run_time=fast * 1)

        sat.clear_updaters()
        sat2.clear_updaters()

        psr_loc2 = [2.5, 1]
        scatterer_location2 = Dot(axes2.c2p(psr_loc2[0], -psr_loc2[1]), color=GREEN_E)
        scatterer_location2.z_index = 3

        self.play(FadeIn(scatterer_location2), FadeOut(cross))
        self.play(
            sat2.animate.move_to(axes2.c2p(0, 0)), sat.animate.move_to(axes.c2p(0, 0))
        )

        u2, graph2, _, connection_line = self.draw_psr(
            psr_loc=psr_loc2, axes=axes, axes2=axes2, color=GREEN_E, sat=sat2
        )
        self.add(graph2)
        sat.add_updater(lambda m: m.move_to(axes.c2p(u2.get_value(), 0)))
        sat2.add_updater(lambda m: m.move_to(axes2.c2p(u2.get_value(), 0)))

        sat2.move_to(axes2.c2p(0, 0))
        sat.move_to(axes.c2p(0, 0))

        cross_size = 0.1
        x_point = np.sqrt((psr_loc2[0] - 0) ** 2 + psr_loc2[1] ** 2)
        point = axes.c2p(0, -x_point)
        cross2 = VGroup(
            Line(
                start=point + LEFT * cross_size + UP * cross_size,
                end=point + RIGHT * cross_size + DOWN * cross_size,
                color=GREEN_D,
            ),
            Line(
                start=point + LEFT * cross_size + DOWN * cross_size,
                end=point + RIGHT * cross_size + UP * cross_size,
                color=GREEN_D,
            ),
        )
        self.play(FadeIn(connection_line))
        self.play(FadeIn(cross2))
        self.play(u2.animate.set_value(4), run_time=5, rate_func=linear)
        self.wait()
        self.play(FadeOut(connection_line, cross2))

        scatterer_locations = [scatterer_location, scatterer_location2]
        for scatterer_loc in [
            [2, 1],
            [2.5, 3],
            [0.5, 2],
            [3.7, 2],
            [0.2, 1],
            [0.8, 1.5],
            [3, 1],
            [1.5, 2.2],
            [3.5, 3],
        ]:
            sat.clear_updaters()
            sat2.clear_updaters()
            psr_loc2 = scatterer_loc
            scatterer_location2 = Dot(
                axes2.c2p(psr_loc2[0], -psr_loc2[1]), color=GREY_D
            )
            scatterer_location2.z_index = 3

            self.play(
                sat2.animate.move_to(axes2.c2p(0, 0)),
                sat.animate.move_to(axes.c2p(0, 0)),
            )

            u2, graph2, _, connection_line = self.draw_psr(
                psr_loc=psr_loc2, axes=axes, axes2=axes2, color=GREY_D, sat=sat2
            )
            self.add(graph2)
            sat.add_updater(lambda m: m.move_to(axes.c2p(u2.get_value(), 0)))
            sat2.add_updater(lambda m: m.move_to(axes2.c2p(u2.get_value(), 0)))

            sat2.move_to(axes2.c2p(0, 0))
            sat.move_to(axes.c2p(0, 0))
            self.play(FadeIn(scatterer_location2, connection_line), run_time=fast * 0.5)
            self.play(u2.animate.set_value(4), run_time=fast * 1, rate_func=linear)
            self.play(FadeOut(connection_line), run_time=fast * 0.5)
            scatterer_locations.append(scatterer_location2)

        self.play(FadeOut(Group(*self.mobjects)))


class Scene6_p2(MovingCameraScene):
    def construct(self):
        self.camera.background_color = "#e6d8bc"
        img_raw = ImageMobject(SAR_IMAGE_PATH).scale(0.4)
        img_processed = ImageMobject(SAR_IMAGE_PATH).scale(0.4)
        img_raw.stretch_to_fit_width(img_processed.width)
        img_raw.stretch_to_fit_height(img_processed.height)

        # Move to center
        img_raw.to_edge(LEFT).shift(RIGHT * 0.45 + 0.25 * DOWN)
        img_processed.to_edge(RIGHT).shift(LEFT * 0.45 + 0.25 * DOWN)

        img_source = (
            Text(
                "Image source: Capella Space, CC BY 4.0 (creativecommons.org/licenses/by/4.0/)",
                font_size=SOURCE_FONT_SIZE,
                font="Zalando Sans",
            )
            .set_color(SOURCE_COLOR)
            .scale(SOURCE_SCALE)
        )

        raw_label = (
            VGroup(
                Text(
                    "Slow Time vs.",
                    font="Zalando Sans SemiExpanded",
                    font_size=SOURCE_FONT_SIZE,
                )
                .set_color(TEXT_COLOR)
                .set_stroke(TEXT_COLOR),
                MathTex(r"\Delta", font_size=SOURCE_FONT_SIZE)
                .set_color(TEXT_COLOR)
                .set_stroke(TEXT_COLOR),
                Text(
                    "TOA", font="Zalando Sans SemiExpanded", font_size=SOURCE_FONT_SIZE
                )
                .set_color(TEXT_COLOR)
                .set_stroke(TEXT_COLOR),
            )
            .scale(0.6)
            .arrange(RIGHT, buff=0.15)
            .next_to(img_raw, UP)
        )
        processed_label = (
            Text(
                "Zero Doppler Coordinates",
                font="Zalando Sans SemiExpanded",
                font_size=SOURCE_FONT_SIZE,
            )
            .scale(0.6)
            .next_to(img_processed, UP)
            .set_color(TEXT_COLOR)
        )
        img_source.to_corner(DL).shift(0.1 * DOWN)
        self.camera.frame.add(img_source)

        # Create a slightly bigger black rectangle as shadow/backdrop
        raw_shadow = create_shadow(
            img_raw, layers=20, scale_factor=1.1, max_opacity=0.1
        )
        processed_shadow = create_shadow(
            img_processed,
            layers=20,
            scale_factor=1.1,
            max_opacity=0.05,
            slant=0.2,
            rotate=0.5,
        ).shift(0.05 * DOWN)
        raw_label.align_to(processed_label, UP)

        self.play(
            FadeIn(raw_shadow, img_raw, img_processed, img_source, processed_shadow)
        )
        self.wait(1)
        self.play(FadeIn(raw_label))
        self.play(FadeIn(processed_label))
        self.wait(2)
        self.camera.frame.save_state()
        left_group = Group(img_raw, raw_label)
        right_group = Group(img_processed, processed_label)
        self.play(
            self.camera.frame.animate.move_to(left_group).set(
                width=left_group.width * 1.9
            ),
            run_time=2,
        )

        self.wait(2)
        self.play(self.camera.frame.animate.move_to(right_group), run_time=2)
        self.wait(1)
        self.play(Restore(self.camera.frame), run_time=2)
        self.wait(2)
        self.play(
            FadeOut(
                raw_label,
                processed_label,
                raw_shadow,
                img_raw,
                img_processed,
                img_source,
                processed_shadow,
            )
        )
