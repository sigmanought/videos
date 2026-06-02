"""SAR pulse transmission, range cells, and time-of-arrival pixel assignment."""

import random
from pathlib import Path

import numpy as np
from manim import *
from PIL import Image
from utils.colors import *
from utils.objects import Clock, Tree
from utils.sat import MeshReflectorSatSideView

fast = 1
CELL_WIDTH = 0.8
SINE_LENGTH = 2 * CELL_WIDTH
CYCLES = 2
FILES_DIR = "./02_bridges_in_stripmap/pngs/"
# add here an image with the color of your background
# and comment everything related to background back in
# otherwise the beams will go "through" the ground
BACKGROUND_RESIZED_PATH = f"{FILES_DIR}/placeholder_resized.png"
BACKGROUND_PATH = f"{FILES_DIR}/placeholder.png"
BACKGROUND_BOTTOM_PATH = f"{FILES_DIR}/placeholder.png"


def sine_func(x):
    return 0.02 * np.sin(1.1 * np.pi * CYCLES * (x / SINE_LENGTH) + 0.15)


def make_beam(path_points, color, stroke_width, dot_scale=0.08):
    temp = VMobject(color=color, stroke_width=stroke_width)
    temp.set_points_as_corners([np.array(p) for p in path_points])
    temp.joint_type = LineJointType.BEVEL
    path = VMobject(color=color, stroke_width=stroke_width)
    path.set_points_as_corners([np.array(p) for p in path_points])
    path.joint_type = LineJointType.BEVEL
    dot = Triangle(color=color, fill_opacity=1).scale(dot_scale).rotate(PI + PI / 4)
    dot.move_to(temp.point_from_proportion(0))
    dot.joint_type = LineJointType.MITER
    return path, dot, temp


def compute_times(traj, speed):
    times = [0]
    cumulative_distance = 0
    for i in range(1, len(traj) - 1):
        seg_len = np.linalg.norm(traj[i] - traj[i - 1])
        cumulative_distance += seg_len
        times.append(cumulative_distance / speed)
    return times


def rotate_dot_manual(dot, times, angles):
    def updater(mob, dt):
        if not hasattr(mob, "elapsed_time"):
            mob.elapsed_time = 0
        mob.elapsed_time += dt
        if not hasattr(mob, "rotated_points"):
            mob.rotated_points = [False] * len(angles)
        if not hasattr(mob, "current_angle"):
            mob.current_angle = 0
        for idx, t in enumerate(times[1:]):
            if not mob.rotated_points[idx] and mob.elapsed_time >= t:
                relative_angle = angles[idx]
                mob.rotate(relative_angle)
                mob.current_angle += relative_angle
                mob.rotated_points[idx] = True
        return mob

    dot.add_updater(updater)
    return dot


class Scene4(MovingCameraScene):
    def add_range_cells_img(self, width=5, grid_xstep=1, height=1):
        range_cells = (
            Rectangle(width=width, height=height, grid_xstep=grid_xstep, grid_ystep=1)
            .set_color(BLACK)
            .shift(2.5 * DOWN)
        )
        return range_cells

    def construct(self):
        self.camera.background_color = "#e6d8bc"
        color_transmit = GREY_C
        color_return = ORANGE
        line_width = 7
        tip_size = 0.2

        # add satellite
        sat = MeshReflectorSatSideView().flip(UP).scale(0.5).shift(5 * LEFT + 2 * UP)

        """bottom_path = Path(BACKGROUND_BOTTOM_PATH)
        if not bottom_path.exists():
            bottom_img = Image.new("RGB", (1920, 540), "#e6d8bc")
            bottom_img.save(BACKGROUND_BOTTOM_PATH)"""

        # ground surface
        surface_pts = [
            [-3, -2, 0],  # start
            [-2, -2.1, 0],
            [-1.5, -2, 0],
            [-0.5, -1.9, 0],
            [0.5, -2, 0],
            [1, -2, 0],
            [1.4, -1.9, 0],
            [2, -2, 0],
            [3.5, -2, 0],  # end
        ]

        surface = VMobject()
        surface.set_points_as_corners([*surface_pts])
        surface.set_color(GRAY_BROWN).set_stroke(width=9.5)
        # background_bottom = ImageMobject(BACKGROUND_BOTTOM_PATH)
        # background_bottom.scale(2.5)
        # scene_height = self.camera.frame_height
        # bottom_y = -scene_height / 2 + background_bottom.height / 2 - 0.29
        # background_bottom.move_to([0, bottom_y, 0])

        start_wave = sat.get_bottom() + 0.05 * LEFT
        wave = Arc(
            radius=0.01,
            start_angle=PI / 2 - PI / 8,
            angle=PI / 4,
            color=color_transmit,
            stroke_width=line_width,
            arc_center=start_wave,
        )
        wave.z_index = 0
        sat.z_index = 1

        # 0:00

        range_arrow = Arrow(
            start=0.5 * LEFT,
            end=0.5 * RIGHT,
            buff=0,
            stroke_width=4,
            tip_length=0.13,
            color=GRAY_BROWN,
        ).scale(0.8)
        range_label = Text(
            "Range",
            font_size=SOURCE_FONT_SIZE,
            color=GRAY_BROWN,
            font="Zalando Sans",
        ).scale(0.5)
        range_group = VGroup(range_arrow, range_label).arrange(RIGHT, buff=0.1)
        range_group.set_z_index(2).to_corner(DL)
        range_group.to_corner(DL).shift(0.2 * DOWN)
        self.play(FadeIn(sat), Create(surface), FadeIn(range_group), run_time=fast * 1)

        self.play(
            wave.animate.scale(100, about_point=start_wave),
            run_time=fast * 1,
            rate_func=linear,
        )
        self.remove(wave)

        # wave from sat to ground
        wave_center = sat.get_center()
        wave_start_angle = -65 * DEGREES  # -PI/3
        wave_angle = PI / 4
        ground_surface_y = -2
        hit_dist = wave_center[1] - ground_surface_y + 0.15

        tracker = ValueTracker(0)
        wave = Arc(
            radius=0.04,
            start_angle=wave_start_angle,
            angle=wave_angle,
            color=color_transmit,
            stroke_width=line_width,
            arc_center=wave_center,
        )
        self.add(wave)

        def wave_updater(m):
            r = tracker.get_value()
            m.become(
                Arc(
                    radius=r,
                    start_angle=wave_start_angle,
                    angle=wave_angle,
                    color=color_transmit,
                    stroke_width=line_width,
                    arc_center=wave_center,
                )
            )

        wave.add_updater(wave_updater)
        self.add(wave)

        # reflections of waves
        wave_centers = [[-2.5, -2, 0], [0, -2.5, 0], [2, -2.5, 0]]
        wave_start_angles = [-PI / 2, -PI / 2, -PI / 2]
        wave_angles = [2 * PI, 2 * PI, 2 * PI]
        trackers = [ValueTracker(0), ValueTracker(0), ValueTracker(0)]
        wave_returns = []
        for idx, (wave_center_i, wave_start_angle_i, wave_angle_i) in enumerate(
            zip(wave_centers, wave_start_angles, wave_angles)
        ):
            wave_i = Arc(
                radius=0.04,
                start_angle=wave_start_angle_i,
                angle=wave_angle_i,
                color=color_return,
                stroke_width=5,
                arc_center=wave_center_i,
            ).set_stroke(width=line_width)
            wave_returns.append(wave_i)

            def wave_updater_i(
                m,
                idx=idx,
                wave_center=wave_center_i,
                wave_start_angle=wave_start_angle_i,
                wave_angle=wave_angle_i,
            ):
                r = trackers[idx].get_value()
                m.become(
                    Arc(
                        radius=r,
                        start_angle=wave_start_angle,
                        angle=wave_angle,
                        color=color_return,
                        stroke_width=line_width,
                        arc_center=wave_center,
                    ).set_stroke(width=line_width)
                )

            wave_i.add_updater(wave_updater_i)
            self.add(wave_i)

        # self.add(background_bottom, surface)
        # wave from feed array to antenna
        speed = hit_dist / 1.2

        # 0:03
        self.play(
            tracker.animate.set_value(hit_dist + 0.3),
            run_time=(hit_dist + 0.3) / speed,
            rate_func=linear,
        )
        dist1 = 1.7
        print((dist1 - 0.3) / speed)
        self.play(
            tracker.animate.set_value(hit_dist + dist1),
            trackers[0].animate.set_value(dist1 - 0.3),
            run_time=(dist1 - 0.3) / speed,
            rate_func=linear,
        )
        dist2 = 0.4
        self.play(
            tracker.animate.set_value(hit_dist + dist1 + dist2),
            trackers[0].animate.set_value(2 - 0.3 + dist2),
            trackers[1].animate.set_value(dist2),
            run_time=(dist2) / speed,
            rate_func=linear,
        )

        dist3 = 1.3
        self.play(
            tracker.animate.set_value(hit_dist + 2 + dist1 + dist2 + dist3),
            trackers[0].animate.set_value(2 - 0.3 + dist2 + dist3),
            trackers[1].animate.set_value(2 - 0.3 + dist3),
            trackers[2].animate.set_value(dist3),
            run_time=(dist3) / speed,
            rate_func=linear,
        )
        self.remove(wave)

        dist4 = 1.3
        self.play(
            trackers[0].animate.set_value(2 - 0.3 + dist2 + dist3 + dist4),
            trackers[1].animate.set_value(2 - 0.3 + dist3 + dist4),
            trackers[2].animate.set_value(2 - 0.3 + dist4),
            run_time=(dist4) / speed,
            rate_func=linear,
        )
        self.remove(wave_returns[0])

        dist5 = 1.45
        self.play(
            trackers[1].animate.set_value(2 - 0.3 + dist3 + dist4 + dist5),
            trackers[2].animate.set_value(2 - 0.3 + dist4 + dist5),
            run_time=(dist5) / speed,
            rate_func=linear,
        )
        self.remove(wave_returns[1])

        dist6 = 3.7
        self.play(
            trackers[2].animate.set_value(2 - 0.3 + dist4 + dist5 + dist6),
            run_time=(dist6) / speed,
            rate_func=linear,
        )
        self.remove(wave_returns[2])

        self.wait(3)
        # 0:09
        pixel = (
            Rectangle(width=1, height=1, color=BLACK)
            .set_fill(opacity=0.8)
            .set_stroke(BLACK)
            .move_to(ORIGIN)
        )
        self.play(FadeIn(pixel))
        # 0:10
        self.wait(2)
        # 0:12
        self.play(pixel.animate.set_fill(WHITE))
        self.wait(1)
        # 0:14
        self.play(FadeOut(pixel))
        # 0:15

        num_pixels = 10
        range_cells = (
            self.add_range_cells_img(width=6.5, grid_xstep=6.5 / num_pixels, height=0.7)
            .shift(0.5 * DOWN + 0.25 * RIGHT)
            .set_fill(GRAY_BROWN, opacity=0.2)
        )
        range_cells.z_index = 102

        colors = [
            BLACK,
            WHITE,
            GREY_D,
            BLACK,
            BLACK,
            WHITE,
            GREY_D,
            WHITE,
            GREY_D,
            BLACK,
        ]
        shift = [
            (i - (num_pixels - 1) / 2) * (6.5 / num_pixels) for i in range(num_pixels)
        ]
        cells = VGroup(
            *[
                Rectangle(width=6.5 / num_pixels, height=0.7)
                .set_fill(color, opacity=0.8)
                .set_stroke(BLACK)
                .shift(shift[i] * RIGHT + 3 * DOWN + 0.25 * RIGHT)
                for i, color in enumerate(colors)
            ]
        )

        self.add(sat)  # add again to preserve z_index

        self.play(
            Create(
                range_cells,
            )
        )
        for submob in range_cells.get_family():
            submob.z_index = 102

        self.play(FadeIn(*cells))
        for cell in cells:
            cell.z_index = 102

        for i in range(7):
            new_colors = random.sample(colors, len(colors))
            self.play(
                *[
                    cell.animate.set_fill(color)
                    for cell, color in zip(cells, new_colors)
                ],
                run_time=0.5,
            )

        range_cells_short = (
            self.add_range_cells_img(width=6.5, grid_xstep=6.5 / 3)
            .shift(0.5 * DOWN + 0.25 * RIGHT)
            .set_fill(color=GRAY_BROWN, opacity=0.2)
        )
        range_cells_short.z_index = 102
        self.play(FadeOut(*cells))
        self.wait(1)
        self.play(Transform(range_cells, range_cells_short))
        for submob in range_cells.get_family():
            submob.z_index = 102
        self.wait(2)

        water2 = FunctionGraph(
            sine_func, x_range=[-3, 3.5], stroke_color=DARK_BLUE, stroke_width=9.5
        ).shift(2 * DOWN)

        origin, target = (sat.get_center(), water2.get_center())
        return_tracker = ValueTracker(0)
        d = np.array(target) - np.array(origin)
        unit = d / np.linalg.norm(d)

        def make_updater(orig, u):
            def updater(m):
                r = return_tracker.get_value()
                end = np.array(orig) + r * u
                arrow = Arrow(
                    orig,
                    end,
                    color=color_transmit,
                    buff=0,
                    stroke_width=line_width,
                    tip_length=tip_size,
                )
                m.become(arrow)

            return updater

        tip = Arrow(color=color_transmit, buff=0, tip_length=tip_size)
        tip.add_updater(make_updater(origin, unit))

        target2, origin2 = sat.get_center(), water2.get_center()
        return_tracker2 = ValueTracker(0)
        d = np.array(target2) - np.array(origin2)
        unit2 = d / np.linalg.norm(d)

        def make_updater(orig, u, target):
            dist = np.linalg.norm(np.array(target) - np.array(orig))

            def updater(m):
                r = return_tracker2.get_value()
                end = np.array(orig) + min(r, dist) * u
                arrow = Arrow(
                    orig,
                    end,
                    color=ORANGE,
                    buff=0,
                    stroke_width=line_width,
                    tip_length=tip_size,
                )
                m.become(arrow)

            return updater

        tip2 = Arrow(color=color_return, buff=0, tip_length=tip_size)
        tip2.add_updater(make_updater(origin2, unit2, sat.get_center()))

        self.add(tip, tip2)
        # self.add(background_bottom, surface)
        self.play(
            return_tracker.animate.set_value(np.linalg.norm(d) + 0.2),
            run_time=3,
            rate_func=linear,
        )
        tip.clear_updaters()
        self.play(
            return_tracker2.animate.set_value(np.linalg.norm(d)),
            run_time=3,
            rate_func=linear,
        )
        tip2.clear_updaters()
        self.remove(tip)
        self.play(FadeOut(tip2))
        self.remove(tip, tip2)

        sat_pos_line = DashedLine(
            start=[sat.get_center()[0], -2, 0] + 0.1 * RIGHT,
            end=sat.get_center() + 0.1 * RIGHT,
            color=BLACK,
            dash_length=0.1,
            dashed_ratio=0.5,
        )
        sat_pos_label = MathTex(
            r"\textsf{(x,y,z)}", color=BLACK, font_size=SOURCE_FONT_SIZE
        ).next_to(sat_pos_line, RIGHT)

        self.play(Create(sat_pos_line))
        self.play(FadeIn(sat_pos_label))
        self.wait(1)
        self.play(FadeOut(sat_pos_label, sat_pos_line))

        # text next to each ray
        range1 = (
            MathTex(
                r"\textsf{r}_\textsf{1}",
                color=color_transmit,
                font_size=SOURCE_FONT_SIZE,
            )
            .scale(SOURCE_SCALE * 3.75)
            .move_to(3 * LEFT + DOWN)
        )
        range2 = (
            MathTex(
                r"\textsf{r}_\textsf{2}",
                color=color_transmit,
                font_size=SOURCE_FONT_SIZE,
            )
            .scale(SOURCE_SCALE * 3.75)
            .move_to(0.5 * LEFT + DOWN)
        )
        range3 = (
            MathTex(
                r"\textsf{r}_\textsf{3}",
                color=color_transmit,
                font_size=SOURCE_FONT_SIZE,
            )
            .scale(SOURCE_SCALE * 3.75)
            .move_to(2 * RIGHT + DOWN)
        )

        targets = [
            (sat.get_center(), color_transmit, water2.get_center()),
            (sat.get_center(), color_transmit, water2.get_left() + 0.1 * RIGHT),
            (sat.get_center(), color_transmit, water2.get_right() + 0.1 * LEFT),
        ]
        tips = []
        return_tracker = ValueTracker(0)
        for origin, color, target in targets:
            d = np.array(target) - np.array(origin)
            unit = d / np.linalg.norm(d)

            def make_updater(orig, u, col):
                def updater(m):
                    r = return_tracker.get_value()
                    end = np.array(orig) + r * u
                    m.become(
                        Arrow(
                            orig,
                            end,
                            color=col,
                            buff=0,
                            stroke_width=line_width,
                            tip_length=0,
                        )
                    )

                return updater

            tip = Arrow(color=color, buff=0, tip_length=tip_size)
            tip.add_updater(make_updater(origin, unit, color))
            tips.append(tip)

        self.add(*tips)
        # self.add(background_bottom, surface)

        self.play(return_tracker.animate.set_value(hit_dist + 5.1 + 0.5), run_time=3)
        self.wait(1)
        self.play(FadeIn(range1))
        self.play(FadeIn(range2))
        self.play(FadeIn(range3))
        # 00:43
        self.wait(3)
        # 00:46

        delta_t = MathTex(r"\Delta \textsf{t} = ", color=BLACK)
        delta_t.move_to(range_cells.get_left() + 0.9 * LEFT)
        delta_t.set_z_index(102)
        self.play(FadeIn(delta_t), FadeOut(range_group))
        delta_t.set_z_index(102)

        # delta t in range cells
        range1_eq = MathTex(
            r"\textsf{2} \cdot",
            r"\textsf{r}_\textsf{1}",
            r"\over \textsf{c}",
            color=BLACK,
        ).scale(0.7)
        range1_eq[1].set_color(GREEN_D)
        range2_eq = MathTex(
            r"\textsf{2} \cdot",
            r"\textsf{r}_\textsf{2}",
            r"\over \textsf{c}",
            color=BLACK,
        ).scale(0.7)
        range2_eq[1].set_color(RED_D)
        range3_eq = MathTex(
            r"\textsf{2} \cdot",
            r"\textsf{r}_\textsf{3}",
            r"\over \textsf{c}",
            color=BLACK,
        ).scale(0.7)
        range3_eq[1].set_color(PURPLE_D)

        range1_eq.move_to(range_cells.get_left() + 1.1 * RIGHT)
        range2_eq.move_to(range_cells.get_center())
        range3_eq.move_to(range_cells.get_right() + 1.1 * LEFT)

        for tip in tips:
            tip.clear_updaters()

        for range_text, range_eq, tips_idx, tips_colors, wait_time1, wait_time2 in zip(
            [range1, range2, range3],
            [range1_eq, range2_eq, range3_eq],
            [1, 0, 2],
            [GREEN_D, RED_D, PURPLE_D],
            [5, 2, 1],
            [5, 2, 2],
        ):
            range_eq.set_z_sindex(102)
            sat.z_index = 500
            self.play(
                tips[tips_idx].animate.set_color(tips_colors),
                range_text.animate.set_color(tips_colors),
            )
            self.wait(wait_time1)
            sat.z_index = 500
            self.play(FadeIn(range_eq))
            self.wait(wait_time2)
            range_eq.set_z_index(103)

        line_eq = Line(
            start=sat.get_center(),
            end=[-2.9, -2, 0],
            color=GREEN_D,
            stroke_width=line_width,
        )
        self.wait(3)
        self.add(line_eq)
        new_end = line_eq.get_start() + 1.15 * rotate_vector(
            line_eq.get_vector(), 28.5 * DEGREES
        )

        tree = Tree(color=GRAY_BROWN).scale(0.4).shift(0.68 * DOWN + 0.47 * LEFT)

        self.play(
            FadeOut(range2),
            tips[0].animate.set_opacity(0),
            line_eq.animate.put_start_and_end_on(line_eq.get_start(), new_end),
            run_time=2,
        )

        # 1:17
        self.play(FadeIn(tree))

        self.wait(16)
        # 1:33
        self.play(FadeOut(range1_eq, range2_eq, range3_eq, delta_t))
        # 1:34
        num_pixels = 3
        colors = [BLACK, WHITE, GREY_D]
        shift = [
            (i - (num_pixels - 1) / 2) * (6.5 / num_pixels) for i in range(num_pixels)
        ]
        cells = VGroup(
            *[
                Rectangle(width=6.5 / num_pixels, height=1)
                .set_fill(color, opacity=0.8)
                .set_stroke(BLACK)
                .shift(shift[i] * RIGHT + 3 * DOWN + 0.25 * RIGHT)
                for i, color in enumerate(colors)
            ]
        )
        for cell in cells:
            cell.set_z_index(102)
        self.play(FadeIn(cells[0]), run_time=1)
        # 1:35
        self.wait(4)
        for cell in cells:
            cell.set_z_index(102)
        self.play(FadeIn(cells[1]), run_time=1)
        # 1:40
        self.wait(6)
        # 1:46
        for cell in cells:
            cell.set_z_index(102)
        self.play(cells[0].animate.set_fill(WHITE, opacity=0.8))
        # 1:47
        self.wait(3)
        self.play(cells[1].animate.set_fill(BLACK, opacity=0.8))
        # 1:50
        self.wait(3)
        # 1:53
        self.play(FadeOut(cells[0], cells[1], *range_cells, line_eq))
        # 1:54
        self.remove(*tips, range1, range2, range3)

        sat_pos = np.array([-3.5, 2, 0])
        self.play(
            sat.animate.scale(0.7).move_to(sat_pos),
            surface.animate.shift(0.5 * UP),
            tree.animate.shift(0.75 * UP).scale(1.5),
        )

        tree_top = tree.get_top()
        ref_pt = tree_top + 0.5 * DOWN + 0.5 * LEFT

        traj1 = [
            sat_pos,
            tree_top + 0 * DOWN + 0.25 * LEFT,
            ref_pt,
            ref_pt + 0.25 * UP + 0.5 * LEFT,
        ]
        traj2 = [
            sat_pos,
            tree_top + 0.5 * DOWN + 0.05 * LEFT,
            tree_top + 0.75 * DOWN + 0.5 * LEFT,
            tree_top + 0.75 * DOWN + 0.5 * LEFT + 0.75 * UP + 0.5 * LEFT,
        ]
        traj3 = [
            sat_pos,
            tree_top + 0.2 * RIGHT + 0.05 * DOWN,
            tree_top + 0.25 * UP + 0.5 * RIGHT,
        ]
        _, dot1, curve1 = make_beam(
            traj1, color_transmit, line_width * 0.6, dot_scale=0.05
        )
        _, dot2, curve2 = make_beam(
            traj2, color_transmit, line_width * 0.6, dot_scale=0.05
        )
        _, dot3, curve3 = make_beam(
            traj3, color_transmit, line_width * 0.6, dot_scale=0.05
        )

        speed = 2
        trail1 = TracedPath(
            dot1.get_center, stroke_color=color_transmit, stroke_width=line_width * 0.6
        )
        trail2 = TracedPath(
            dot2.get_center, stroke_color=color_transmit, stroke_width=line_width * 0.6
        )
        trail3 = TracedPath(
            dot3.get_center, stroke_color=color_transmit, stroke_width=line_width * 0.6
        )
        self.add(trail1, trail2, trail3)
        dot1.z_index = 0
        dot2.z_index = 0
        dot3.z_index = 0
        sat.z_index = 1

        self.add(dot1, dot2, dot3)
        self.add(sat)

        times1 = compute_times(traj1, speed)
        times2 = compute_times(traj2, speed)
        times3 = compute_times(traj3, speed)

        angles1 = [-PI / 2, -PI / 2, 0]  # example
        angles2 = [12 * DEGREES, 23 * DEGREES, 0]
        angles3 = [PI / 2.5, 0]

        rotate_dot_manual(dot1, times1, angles1)
        rotate_dot_manual(dot2, times2, angles2)
        rotate_dot_manual(dot3, times3, angles3)

        self.play(
            MoveAlongPath(
                dot1, curve1, run_time=curve1.get_arc_length() / speed, rate_func=linear
            ),
            MoveAlongPath(
                dot2, curve2, run_time=curve2.get_arc_length() / speed, rate_func=linear
            ),
            MoveAlongPath(
                dot3, curve3, run_time=curve3.get_arc_length() / speed, rate_func=linear
            ),
        )
        self.add(curve1, curve2, curve3)
        self.wait(2)
        self.play(
            FadeOut(
                dot1,
                dot2,
                dot3,
                trail1,
                trail2,
                trail3,
                curve1,
                curve2,
                curve3,
                surface,
                sat,
                tree,
            )
        )

        # now animation -> pixel = time of arrival
        pixel = (
            Rectangle(width=1.5, height=1.5, color=BLACK)
            .set_fill(opacity=0.8)
            .set_stroke(BLACK)
            .move_to(ORIGIN + 2.5 * LEFT)
        )
        self.play(FadeIn(pixel))

        clock = (
            Clock(radius=0.75).set_stroke(BLACK, width=5).move_to(ORIGIN + 2.5 * RIGHT)
        )

        arrow1 = CurvedArrow(
            start_point=pixel.get_right() + 0.75 * RIGHT,
            end_point=clock.get_left() + 0.75 * LEFT,
            angle=-PI / 8,
            color=BLACK,
        ).shift(0.5 * UP)

        arrow2 = CurvedArrow(
            start_point=clock.get_left() + 0.75 * LEFT,
            end_point=pixel.get_right() + 0.75 * RIGHT,
            angle=-PI / 8,
            color=BLACK,
        ).shift(0.5 * DOWN)

        self.play(FadeIn(arrow1))
        self.play(FadeIn(clock))
        self.play(FadeIn(arrow2))
        self.play(
            Succession(Indicate(pixel, color=BLACK), Indicate(clock, color=BLACK))
        )
        self.play(FadeOut(pixel, clock, arrow1, arrow2))

        # add water
        water2 = FunctionGraph(
            sine_func, x_range=[-3, 3.5], stroke_color=DARK_BLUE, stroke_width=9.5
        ).shift(2 * DOWN)

        sat = MeshReflectorSatSideView().flip(UP).scale(0.5).shift(5 * LEFT + 2 * UP)
        self.play(FadeIn(sat), Create(water2), run_time=1)
        water2.set_z_index(11)
        # background_bottom.set_z_index(10)

        # self.add(background_bottom)

        legend = VGroup(
            VGroup(
                Dot(color=GREY, radius=0.15),
                Text(
                    "transmit",
                    font="Zalando Sans",
                    font_size=SOURCE_FONT_SIZE,
                    color=BLACK,
                ).scale(SOURCE_SCALE * 2),
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Dot(color=ORANGE),
                Text(
                    "echo", font="Zalando Sans", font_size=SOURCE_FONT_SIZE, color=BLACK
                ).scale(SOURCE_SCALE * 2),
            ).arrange(RIGHT, buff=0.2),
        ).arrange(RIGHT, buff=0.8)
        legend.set_z_index(100)

        legend.to_edge(DOWN).shift(0.5 * UP)
        self.play(FadeIn(legend))

        color_transmit = GREY_C
        color_return = ORANGE
        ### first transmit and reflect away from water with arcs
        start_wave = sat.get_bottom() + 0.05 * LEFT
        wave = Arc(
            radius=0.01,
            start_angle=PI / 2 - PI / 8,
            angle=PI / 4,
            color=color_transmit,
            arc_center=start_wave,
            stroke_width=line_width,
        )
        wave.z_index = 0
        sat.z_index = 1
        self.play(
            wave.animate.scale(100, about_point=start_wave),
            run_time=1,
            rate_func=linear,
        )
        self.remove(wave)

        # incoming wave setup
        wave_center = sat.get_center()
        wave_start_angle = -65 * DEGREES
        wave_angle = PI / 4

        water_y = -2
        hit_dist = wave_center[1] - water_y + 0.15

        # reflected origin: mirror sat over water surface
        reflected_origin = [sat.get_center()[0], -sat.get_center()[1] - 4, 0]
        dot = Dot(reflected_origin, color=RED)
        self.add(dot)

        # incoming wave
        tracker = ValueTracker(0)
        wave = Arc(
            radius=0.04,
            start_angle=wave_start_angle,
            angle=wave_angle,
            color=color_transmit,
            arc_center=wave_center,
            stroke_width=line_width,
        )
        self.add(wave)

        def wave_updater(m):
            r = tracker.get_value()
            m.become(
                Arc(
                    radius=r,
                    start_angle=wave_start_angle,
                    angle=wave_angle,
                    color=color_transmit,
                    arc_center=wave_center,
                    stroke_width=line_width,
                )
            )

        wave.add_updater(wave_updater)
        self.add(wave)

        # return wave starts at hit_dist so it begins exactly at the water surface
        return_tracker = ValueTracker(0)
        return_wave = Arc(
            radius=0.04,
            start_angle=0,
            angle=PI / 2.7,
            color=color_return,
            arc_center=reflected_origin,
            stroke_width=line_width,
        )

        def return_updater(m):
            r = return_tracker.get_value()
            stroke_width = 0 if r < hit_dist + 0.2 else line_width
            m.become(
                Arc(
                    radius=r,
                    start_angle=0,
                    angle=PI / 2.7,
                    color=color_return,
                    arc_center=reflected_origin,
                    stroke_width=stroke_width,
                )
            )

        return_wave.add_updater(return_updater)
        self.add(return_wave)

        speed = (hit_dist) / 2
        self.play(
            tracker.animate.set_value(hit_dist + 0.1),
            return_tracker.animate.set_value(hit_dist + 0.1),
            run_time=2,
            rate_func=linear,
        )

        self.play(
            tracker.animate.set_value(hit_dist + 5.1),
            return_tracker.animate.set_value(hit_dist + 5.1),
            run_time=(hit_dist + 5.5) / speed,
            rate_func=linear,
        )

        wave.remove_updater(wave_updater)
        self.remove(wave)
        return_wave.clear_updaters()
        self.remove(return_wave)

        ### second transmit with arrows
        # add back waves
        # incoming wave
        tracker = ValueTracker(0)
        wave = Arc(
            radius=0.04,
            start_angle=wave_start_angle,
            angle=wave_angle,
            color=color_transmit,
            stroke_width=line_width,
            arc_center=wave_center,
        ).set_stroke(opacity=0.5)
        self.add(wave)

        def wave_updater(m):
            r = tracker.get_value()
            m.become(
                Arc(
                    radius=r,
                    start_angle=wave_start_angle,
                    angle=wave_angle,
                    color=color_transmit,
                    stroke_width=line_width,
                    arc_center=wave_center,
                ).set_stroke(opacity=0.5)
            )

        wave.add_updater(wave_updater)
        self.add(wave)

        # return wave starts at hit_dist so it begins exactly at the water surface
        return_tracker = ValueTracker(0)
        return_wave = Arc(
            radius=0.04,
            start_angle=0,
            angle=PI / 2.7,
            color=color_return,
            stroke_width=line_width,
            arc_center=reflected_origin,
        ).set_stroke(opacity=0.5)

        def return_updater(m):
            r = return_tracker.get_value()
            stroke_width = 0 if r < hit_dist + 0.2 else line_width
            m.become(
                Arc(
                    radius=r,
                    start_angle=0,
                    angle=PI / 2.7,
                    color=color_return,
                    stroke_width=stroke_width,
                    arc_center=reflected_origin,
                ).set_stroke(opacity=0.5)
            )

        return_wave.add_updater(return_updater)
        self.add(return_wave)

        targets = [
            (sat.get_center(), color_transmit, water2.get_center()),
            (sat.get_center(), color_transmit, water2.get_left() + 0.1 * RIGHT),
            (sat.get_center(), color_transmit, water2.get_right() + 0.1 * LEFT),
            (reflected_origin, color_return, water2.get_center()),
            (reflected_origin, color_return, water2.get_left() + 0.1 * RIGHT),
            (reflected_origin, color_return, water2.get_right() + 0.1 * LEFT),
        ]

        tips = []
        for origin, color, target in targets:
            d = np.array(target) - np.array(origin)
            unit = d / np.linalg.norm(d)

            def make_updater(orig, u, col):
                def updater(m):
                    r = return_tracker.get_value()
                    end = np.array(orig) + r * u
                    m.become(
                        Arrow(
                            orig,
                            end,
                            color=col,
                            buff=0,
                            stroke_width=line_width,
                            tip_length=tip_size,
                        )
                    )

                return updater

            tip = Arrow(color=color, buff=0, tip_length=tip_size)
            tip.add_updater(make_updater(origin, unit, color))
            tips.append(tip)

        self.add(*tips)

        self.play(
            tracker.animate.set_value(hit_dist + 5.1),
            return_tracker.animate.set_value(hit_dist + 5.1),
            run_time=(hit_dist + 5.5) / speed,
            rate_func=linear,
        )

        wave.remove_updater(wave_updater)
        self.remove(wave)
        self.play(
            return_tracker.animate.set_value(hit_dist + 5.1 + 0.5),
            run_time=(0.5) / speed,
            rate_func=linear,
        )
        return_wave.clear_updaters()
        self.remove(return_wave)
        self.play(FadeOut(legend))
        for tip in tips:
            tip.clear_updaters()

        range_cells = self.add_range_cells_img(width=6.5, grid_xstep=6.5 / 3).shift(
            0.5 * DOWN
        )
        range_cells.z_index = 102
        self.add(sat)  # add again to preserve z_index

        self.play(FadeIn(range_cells))
        for submob in range_cells.get_family():
            submob.z_index = 102

        self.play(range_cells.animate.set_fill(color=BLACK, opacity=0.7))
        self.wait(2)
