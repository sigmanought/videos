"""Single, double, and triple bounce on the bridge."""

from pathlib import Path

import manim
import numpy as np
from manim import *
from numpy.linalg import norm
from PIL import Image
from utils.animations import draw_angle
from utils.colors import *
from utils.decoration import create_shadow
from utils.objects import Cloud
from utils.sat import MeshReflectorSat, MeshReflectorSatSideView

fast = 1
RES_TYPE = "high"
IMG_WIDTH = 6
IMG_HEIGHT = 4
CELL_WIDTH = 0.8
SINE_LENGTH = 2 * CELL_WIDTH
CYCLES = 2
SAR_FILES_DIR = "./sar_bridges/scenes/files"
FILES_DIR = "./02_bridges_in_stripmap/pngs/"
GOLDEN_GATE_PATH = f"{FILES_DIR}/golden_gate.svg"
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


class Scene5(MovingCameraScene):
    def add_range_cells_img(self, width=5, grid_xstep=1, height=1):
        range_cells = (
            Rectangle(width=width, height=height, grid_xstep=grid_xstep, grid_ystep=1)
            .set_color(BLACK)
            .shift(2.5 * DOWN)
        )
        return range_cells

    def construct(self):
        self.camera.background_color = "#e6d8bc"
        line_width = 12
        # add satellite
        sat = MeshReflectorSatSideView().flip(UP).scale(0.5).shift(5 * LEFT + 2 * UP)

        bottom_path = Path(BACKGROUND_BOTTOM_PATH)
        if not bottom_path.exists():
            img = Image.open(BACKGROUND_PATH)
            w, h = img.size
            water_frac = 0.73
            crop_y = int(h * water_frac)
            bottom_img = img.crop((0, crop_y, w, h))
            bottom_img.save(BACKGROUND_BOTTOM_PATH)

        # add water
        water2 = FunctionGraph(
            sine_func, x_range=[-3.2, 3.7], stroke_color=DARK_BLUE, stroke_width=7
        ).shift(2.5 * DOWN)

        golden_gate = SVGMobject(GOLDEN_GATE_PATH, fill_opacity=1)
        golden_gate.set_color(INTERNATIONAL_ORANGE).set_stroke(DARK_RED, width=5)
        golden_gate.scale(2.7).shift(2.5 * RIGHT)

        self.play(FadeIn(golden_gate))
        # add bridge, fade out towers and pillars
        bridge1 = (
            Rectangle(
                height=0.195,
                width=1.175,
                color=INTERNATIONAL_ORANGE,
                fill_color=INTERNATIONAL_ORANGE,
                fill_opacity=1,
            )
            .set_stroke(DARK_RED, width=5)
            .shift(2.5 * RIGHT + 1.205 * DOWN)
        )

        self.add(bridge1)
        self.wait()

        water1 = FunctionGraph(
            sine_func, x_range=[-5.5, 5.5], stroke_color=DARK_BLUE, stroke_width=12
        ).shift(2.72 * DOWN)

        self.play(Create(water1))
        self.play(FadeIn(sat))
        self.wait(1)
        water1.set_z_index(2)
        self.add(water1)
        self.wait(1)

        bridge2 = (
            Rectangle(
                height=0.3,
                width=3,
                color=INTERNATIONAL_ORANGE,
                fill_color=INTERNATIONAL_ORANGE,
                fill_opacity=1,
            )
            .shift(1.5 * RIGHT + 0.5 * DOWN)
            .set_stroke(DARK_RED, width=5)
        )
        bridge2.set_z_index = 2

        self.play(golden_gate.animate.set_opacity(0.3))
        self.wait(1)
        self.play(FadeOut(golden_gate))
        self.play(
            AnimationGroup(
                Transform(water1, water2),
                Transform(bridge1, bridge2),
                sat.animate.shift(0.7 * UP).scale(0.7),
            )
        )
        self.play(sat.animate.set_fill(opacity=0).set_stroke(opacity=0))
        sat.shift(0.5 * RIGHT + UP)
        self.wait(1)
        bridge1.z_index = 3

        direction_db = bridge1.get_left() - sat.get_center()
        start = bridge1.get_left() + 0.1 * DOWN
        traj_sb = [
            sat.get_center() + 0.1 * UP + direction_db * 0.3,
            bridge1.get_left() + 0.1 * UP,
            sat.get_center() + 0.1 * UP + direction_db * 0.3,
        ]
        unit_dir = normalize(direction_db) * np.array([-1, 1, 0])
        for t in np.linspace(0, 20, 100000):
            point = start + t * unit_dir
            if (
                water1.get_left()[0] <= point[0] <= water1.get_right()[0]
                and water1.get_bottom()[1] <= point[1] <= water1.get_top()[1]
            ):
                distance_water_bridge_db = t
                break

        traj_db = [
            sat.get_center() + 0.1 * DOWN + direction_db * 0.3,
            bridge1.get_left() + 0.1 * DOWN,
            bridge1.get_left() + 0.1 * DOWN + unit_dir * distance_water_bridge_db,
            water1.get_left() + 1.2 * RIGHT - direction_db * 0.6,
        ]

        # Cast a ray and find where it hits bridge1
        start = water1.get_left() + 2 * RIGHT
        unit_dir = normalize(direction_db) * np.array([1, -1, 0])
        for t in np.linspace(0, 20, 100000):
            point = start + t * unit_dir
            if (
                bridge1.get_left()[0] <= point[0] <= bridge1.get_right()[0]
                and bridge1.get_bottom()[1] <= point[1] <= bridge1.get_top()[1]
            ):
                distance_water_bridge_tb = t
                break

        traj_tb = [
            water1.get_left() + 2.1 * RIGHT - direction_db * 0.72,
            water1.get_left() + 2 * RIGHT + 0.05 * UP,
            water1.get_left() + 2 * RIGHT + distance_water_bridge_tb * unit_dir,
            water1.get_left() + 2 * RIGHT + 0.05 * UP,
            water1.get_left() + 2.1 * RIGHT - direction_db * 0.72,
        ]

        _, dot_sb, curve_sb = make_beam(traj_sb, COLOR_SB, line_width * 0.5)
        _, dot_db, curve_db = make_beam(traj_db, COLOR_DB, line_width * 0.5)
        _, dot_tb, curve_tb = make_beam(traj_tb, COLOR_TB, line_width * 0.5)

        triggered_sb = False
        sb_finished = False

        def updater_sb(m):
            nonlocal triggered_sb
            nonlocal sb_finished

            if not triggered_sb and np.linalg.norm(m.get_center() - traj_sb[1]) < 0.1:
                m.rotate(PI)
                triggered_sb = True
            if triggered_sb and not sb_finished:
                if np.linalg.norm(m.get_center() - traj_sb[2]) < 0.1:
                    sb_finished = True

        triggered_db_1 = False
        triggered_db_2 = False
        db_finished = False

        def updater_db(m):
            nonlocal triggered_db_1
            nonlocal triggered_db_2
            nonlocal db_finished
            if not triggered_db_1 and np.linalg.norm(m.get_center() - traj_db[1]) < 0.1:
                p0, p1, p2 = traj_db[0], traj_db[1], traj_db[2]
                v1 = p1 - p0
                v2 = p2 - p1
                angle = PI - np.arccos(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                )
                m.rotate(angle + PI)
                triggered_db_1 = True

            if not triggered_db_2 and np.linalg.norm(m.get_center() - traj_db[2]) < 0.1:
                p0, p1, p2 = traj_db[1], traj_db[2], traj_db[3]
                v1 = p1 - p0
                v2 = p2 - p1
                angle = np.arccos(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                )
                m.rotate(angle + PI)
                triggered_db_2 = True

            if triggered_db_2 and triggered_db_1 and not db_finished:
                if np.linalg.norm(m.get_center() - traj_db[3]) < 0.1:
                    db_finished = True
            if db_finished:
                return

        triggered_tb_1 = False
        triggered_tb_2 = False
        triggered_tb_3 = False
        tb_finished = False

        def updater_tb(m):
            nonlocal triggered_tb_1
            nonlocal triggered_tb_2
            nonlocal triggered_tb_3
            nonlocal tb_finished
            if (
                not triggered_tb_1
                and np.linalg.norm(m.get_center() - traj_tb[1]) < 0.05
            ):
                p0, p1, p2 = traj_tb[0], traj_tb[1], traj_tb[2]
                v1 = p1 - p0
                v2 = p2 - p1
                angle = np.arccos(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                )
                m.rotate(angle + 2 * PI)
                triggered_tb_1 = True

            if (
                not triggered_tb_2
                and np.linalg.norm(m.get_center() - traj_tb[2]) < 0.05
            ):
                m.rotate(PI)
                triggered_tb_2 = True

            if (
                not triggered_tb_3
                and triggered_tb_1
                and triggered_tb_2
                and np.linalg.norm(m.get_center() - traj_tb[3]) < 0.05
            ):
                p0, p1, p2 = traj_tb[0], traj_tb[1], traj_tb[2]
                v1 = p1 - p0
                v2 = p2 - p1
                angle = np.arccos(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                )
                m.rotate(-(angle + 2 * PI))
                triggered_tb_3 = True

            if triggered_tb_1 and triggered_tb_2 and triggered_tb_3 and not tb_finished:
                tb_finished = True

        dot_sb.add_updater(updater_sb)
        dot_sb.set_z_index = 0
        dot_db.add_updater(updater_db)
        dot_db.set_z_index = 0
        dot_tb.add_updater(updater_tb)
        dot_tb.set_z_index = 0

        speed = 2
        trail_sb = TracedPath(
            dot_sb.get_center, stroke_color=COLOR_SB, stroke_width=line_width * 0.5
        )
        trail_db = TracedPath(
            dot_db.get_center, stroke_color=COLOR_DB, stroke_width=line_width * 0.5
        )
        trail_tb = TracedPath(
            dot_tb.get_center, stroke_color=COLOR_TB, stroke_width=line_width * 0.5
        )

        trail_sb.z_index = 0
        trail_db.z_index = 0
        trail_tb.z_index = 1
        self.camera.frame.save_state()
        self.add(trail_sb, trail_db, trail_tb)
        self.play(
            MoveAlongPath(
                dot_sb,
                curve_sb,
                run_time=curve_sb.get_arc_length() / speed,
                rate_func=linear,
            )
        )
        self.wait(3)
        curve_sb.z_index = 0
        self.add(curve_sb)
        runtime = curve_db.get_arc_length() / speed
        self.play(MoveAlongPath(dot_db, curve_db, run_time=runtime, rate_func=linear))
        self.remove(trail_db)
        curve_db.z_index = 0
        self.add(curve_db)
        self.wait(7 - runtime)
        runtime = curve_tb.get_arc_length() / speed
        self.play(MoveAlongPath(dot_tb, curve_tb, run_time=runtime, rate_func=linear))
        curve_tb.z_index = 1
        self.add(curve_tb)
        self.wait(11 - runtime)

        # add range cells
        self.play(self.camera.frame.animate.shift(DOWN * 0.7).scale(1.05), runtime=1)

        num_pixels = 7
        range_cells = self.add_range_cells_img(
            width=6.5, grid_xstep=6.5 / num_pixels, height=0.75
        ).shift(DOWN + 0.25 * RIGHT)
        self.play(FadeIn(range_cells), runtime=1)

        range_arrow = Arrow(
            start=0.75 * LEFT,
            end=0.75 * RIGHT,
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
        range_group.next_to(range_cells, DOWN, buff=0.3)
        self.play(
            FadeIn(range_group),
            self.camera.frame.animate.shift(0.25 * DOWN),
            run_time=1,
        )
        self.wait(1)
        self.play(
            FadeOut(range_group), self.camera.frame.animate.shift(0.25 * UP), run_time=1
        )

        scale_factor_dot = 1.5
        self.play(
            curve_sb.animate.set_stroke(width=line_width),
            dot_sb.animate.scale(scale_factor_dot),
            run_time=0.5,
        )
        self.play(
            curve_sb.animate.set_stroke(width=line_width * 0.5),
            dot_sb.animate.scale(1 / scale_factor_dot),
            run_time=0.5,
        )
        self.play(
            curve_db.animate.set_stroke(width=line_width),
            dot_db.animate.scale(scale_factor_dot),
            run_time=0.5,
        )
        self.play(
            curve_db.animate.set_stroke(width=line_width * 0.5),
            dot_db.animate.scale(1 / scale_factor_dot),
            run_time=0.5,
        )
        self.play(
            curve_tb.animate.set_stroke(width=line_width),
            dot_tb.animate.scale(scale_factor_dot),
            run_time=0.5,
        )
        self.play(
            curve_tb.animate.set_stroke(width=line_width * 0.5),
            dot_tb.animate.scale(1 / scale_factor_dot),
            run_time=0.5,
        )

        self.play(range_cells.animate.scale(1.1), run_time=1)
        self.play(range_cells.animate.scale(1 / 1.1), run_time=1)
        self.wait(1)

        labels = [
            r"7\textsf{s}",
            r"8\textsf{s}",
            r"9\textsf{s}",
            r"10\textsf{s}",
            r"11\textsf{s}",
            r"12\textsf{s}",
            r"13\textsf{s}",
        ]
        cell_width = 6.5 / num_pixels
        left_edge = range_cells.get_left()[0]
        texts = VGroup()
        for i, label in enumerate(labels):
            x = left_edge + cell_width * (i + 0.5)
            txt = MathTex(
                label, font_size=SOURCE_FONT_SIZE, color=SOURCE_COLOR_DARK
            ).scale(1 / 1.4)
            txt.move_to([x, range_cells.get_center()[1], 0])
            texts.add(txt)

        self.play(
            LaggedStart(*[FadeIn(txt) for txt in texts], lag_ratio=0.2), run_time=4
        )
        note = Text(
            "In practice, these time intervals are in nanoseconds.",
            font_size=SOURCE_FONT_SIZE,
            font="Zalando Sans",
            color=SOURCE_COLOR,
        ).scale(0.33)
        note.to_corner(DL).shift(DOWN)
        self.play(FadeIn(note), run_time=1)
        self.wait(1)

        self.play(*[text.animate.scale(1.1) for text in texts], run_time=1)
        self.play(*[text.animate.scale(1 / 1.1) for text in texts], run_time=1)

        self.play(
            FadeOut(
                curve_tb, dot_tb, trail_tb, curve_db, dot_db, curve_sb, dot_sb, trail_sb
            ),
            FadeOut(*texts, note),
            run_time=1,
        )
        self.remove(
            curve_tb, dot_tb, trail_tb, curve_db, dot_db, curve_sb, dot_sb, trail_sb
        )

        _, dot_sb, curve_sb = make_beam(traj_sb, COLOR_SB, line_width * 0.5)
        _, dot_db, curve_db = make_beam(traj_db, COLOR_DB, line_width * 0.5)
        _, dot_tb, curve_tb = make_beam(traj_tb, COLOR_TB, line_width * 0.5)

        dot_sb.add_updater(updater_sb)
        dot_db.add_updater(updater_db)
        dot_tb.add_updater(updater_tb)

        speed = 2
        trail_sb = TracedPath(
            dot_sb.get_center, stroke_color=COLOR_SB, stroke_width=line_width * 0.5
        )
        trail_sb.z_index = 0
        trail_db = TracedPath(
            dot_db.get_center, stroke_color=COLOR_DB, stroke_width=line_width * 0.5
        )
        trail_db.z_index = 0
        trail_tb = TracedPath(
            dot_tb.get_center, stroke_color=COLOR_TB, stroke_width=line_width * 0.5
        )
        trail_tb.z_index = 1

        # add updater for range cell fill
        colors = [BLACK, WHITE, BLACK, WHITE, BLACK, WHITE, BLACK]
        shift = [
            (i - (num_pixels - 1) / 2) * (6.5 / num_pixels) for i in range(num_pixels)
        ]
        cells = VGroup(
            *[
                Rectangle(width=6.5 / num_pixels, height=0.75)
                .set_fill(color, opacity=0)
                .set_stroke(BLACK, opacity=0)
                .shift(shift[i] * RIGHT + 3 * DOWN)
                for i, color in enumerate(colors)
            ]
        ).shift(0.5 * DOWN + 0.25 * RIGHT)
        self.add(cells)

        triggered_sb = False
        triggered_db_1 = False
        triggered_db_2 = False
        triggered_tb_1 = False
        triggered_tb_2 = False
        triggered_tb_3 = False
        sb_finished = False
        db_finished = False
        tb_finished = False

        speed = 1
        print(
            curve_sb.get_arc_length() / speed,
            curve_db.get_arc_length() / speed,
            curve_tb.get_arc_length() / speed,
        )

        run0 = False
        elapsed_time0 = 0

        def updater_range_cell0(m, dt):
            nonlocal run0, elapsed_time0
            elapsed_time0 += dt
            if not run0 and elapsed_time0 >= 7:
                m.set_fill(opacity=1)
                run0 = 1

        run1 = False
        elapsed_time1 = 0

        def updater_range_cell1(m, dt):
            nonlocal run1, elapsed_time1
            elapsed_time1 += dt
            if not run1 and elapsed_time1 >= 8:
                m.set_fill(opacity=1)
                run1 = 1

        run2 = False
        elapsed_time2 = 0

        def updater_range_cell2(m, dt):
            nonlocal run2, elapsed_time2
            elapsed_time2 += dt
            if not run2 and elapsed_time2 >= 9:
                m.set_fill(opacity=1)
                run2 = 1

        run3 = False
        elapsed_time3 = 0

        def updater_range_cell3(m, dt):
            nonlocal run3, elapsed_time3
            elapsed_time3 += dt
            if not run3 and elapsed_time3 >= 10:
                m.set_fill(opacity=1)
                run3 = 1

        run4 = False
        elapsed_time4 = 0

        def updater_range_cell4(m, dt):
            nonlocal run4, elapsed_time4
            elapsed_time4 += dt
            if not run4 and elapsed_time4 >= 11:
                m.set_fill(opacity=1)
                run4 = 1

        run5 = False
        elapsed_time5 = 0

        def updater_range_cell5(m, dt):
            nonlocal run5, elapsed_time5
            elapsed_time5 += dt
            if not run5 and elapsed_time5 >= 12:
                m.set_fill(opacity=1)
                run5 = 1

        run6 = False
        elapsed_time6 = 0

        def updater_range_cell6(m, dt):
            nonlocal run6, elapsed_time6
            elapsed_time6 += dt
            if not run6 and elapsed_time6 >= 13:
                m.set_fill(opacity=1)
                run6 = 1

        cells[0].add_updater(updater_range_cell0)
        cells[1].add_updater(updater_range_cell1)
        cells[2].add_updater(updater_range_cell2)
        cells[3].add_updater(updater_range_cell3)
        cells[4].add_updater(updater_range_cell4)
        cells[5].add_updater(updater_range_cell5)
        cells[6].add_updater(updater_range_cell6)

        self.add(trail_sb, trail_db, trail_tb)
        t = ValueTracker(0)
        cell_width = 6.5 / num_pixels
        left_edge = range_cells.get_left()[0]
        y = range_cells.get_center()[1]
        timer = DecimalNumber(
            0,
            num_decimal_places=2,
            font_size=SOURCE_FONT_SIZE,
            color=SOURCE_COLOR_DARK,
        ).scale(1 / 1.4)
        timer_finished = False

        def update_timer(mob, dt):
            nonlocal timer_finished
            new_t = t.get_value() + dt
            if new_t > 13:
                new_t = 13
            t.set_value(new_t)
            mob.set_value(new_t)
            idx = int(new_t - 7 + 1)
            idx = max(0, min(idx, num_pixels - 1))
            mob.move_to([left_edge + cell_width * (idx + 0.5), y, 0])
            if new_t >= 13 and not timer_finished:
                timer_finished = True
                mob.clear_updaters()
                self.remove(mob)

        timer.add_updater(update_timer)
        self.add(range_cells, timer)

        self.play(
            MoveAlongPath(
                dot_sb,
                curve_sb,
                run_time=curve_sb.get_arc_length() / speed,
                rate_func=linear,
            ),
            MoveAlongPath(
                dot_db,
                curve_db,
                run_time=curve_db.get_arc_length() / speed,
                rate_func=linear,
            ),
            MoveAlongPath(
                dot_tb,
                curve_tb,
                run_time=curve_tb.get_arc_length() / (speed * 1.1),
                rate_func=linear,
            ),
        )
        self.remove(trail_db)
        curve_sb.z_index = 0
        curve_db.z_index = 0
        curve_tb.z_index = 1
        self.add(curve_sb, curve_db, curve_tb)
        self.wait(2)
        # now show SAR image again
        self.play(self.camera.frame.animate.shift(4.5 * RIGHT).scale(1.4))

        img_high = ImageMobject(BACKGROUND_PATH).scale(0.07)
        img_high.stretch_to_fit_width(IMG_WIDTH * 1.4)
        img_high.stretch_to_fit_height(IMG_HEIGHT * 1.4)

        img_sb = ImageMobject(BACKGROUND_PATH).scale(0.07)
        img_sb.stretch_to_fit_width(IMG_WIDTH * 1.4)
        img_sb.stretch_to_fit_height(IMG_HEIGHT * 1.4)

        img_db = ImageMobject(BACKGROUND_PATH).scale(0.07)
        img_db.stretch_to_fit_width(IMG_WIDTH * 1.4)
        img_db.stretch_to_fit_height(IMG_HEIGHT * 1.4)

        img_tb = ImageMobject(BACKGROUND_PATH).scale(0.07)
        img_tb.stretch_to_fit_width(IMG_WIDTH * 1.4)
        img_tb.stretch_to_fit_height(IMG_HEIGHT * 1.4)

        img_source = (
            Text(
                "Image source: Capella Space, CC BY 4.0 (creativecommons.org/licenses/by/4.0/)",
                font_size=SOURCE_FONT_SIZE,
                font="Zalando Sans",
            )
            .set_color(SOURCE_COLOR)
            .scale(SOURCE_SCALE)
        )

        img = ImageMobject(BACKGROUND_PATH).scale(0.07)
        img.stretch_to_fit_width(IMG_WIDTH * 1.4)
        img.stretch_to_fit_height(IMG_HEIGHT * 1.4)
        sar_shadow = create_shadow(img, layers=20, scale_factor=1.1, max_opacity=0.1)
        img.shift(9.2 * RIGHT + 0.65 * DOWN)
        sar_shadow.shift(9.2 * RIGHT + 0.65 * DOWN)
        img_source.next_to(img, DOWN).align_to(img, LEFT)
        self.play(FadeIn(sar_shadow, img, img_source))
        self.wait(4)

        img_sb.move_to(img.get_center())
        self.play(FadeIn(img_sb), run_time=0.5)
        self.wait(1.5)

        img_db.move_to(img.get_center())
        self.play(FadeIn(img_db), run_time=0.5)
        self.wait(1.5)

        img_tb.move_to(img.get_center())
        self.play(FadeIn(img_tb), run_time=0.5)
        self.wait(2)

        img_high.move_to(img.get_center())
        self.play(FadeIn(img_high), run_time=0.5)
        self.remove(img, img_sb, img_db, img_tb)
        self.wait(1)

        self.play(
            self.camera.frame.animate.shift(4.5 * RIGHT).scale(1 / (1.3)),
            bridge1.animate.set_opacity(0),
            *[range_cell.animate.set_opacity(0) for range_cell in range_cells],
            *[cell.animate.set_opacity(0) for cell in cells],
            water1.animate.set_opacity(0),
            run_time=1,
        )
        self.wait(4)
        self.play(FadeIn(img), run_time=1)
        self.remove(img_high)
        self.wait(2)

        sat = (
            MeshReflectorSat(radius=2, num_scallops=12, scallop_depth=0.7)
            .set_stroke(1)
            .scale(0.2)
            .rotate(PI)
        )
        sat.next_to(img_high, UP).align_to(img_high, LEFT).shift(0.1 * UP)
        self.play(self.camera.frame.animate.shift(0.5 * UP), run_time=1)
        self.play(FadeIn(sat), run_time=1)
        self.play(
            sat.animate.shift(IMG_WIDTH * 1.4 * RIGHT), run_time=4, rate_func=linear
        )

        self.wait(4)
        self.play(
            FadeOut(sar_shadow, img, *cells, range_cells, sat, img_source), run_time=1
        )


class Scene5_p2(MovingCameraScene):
    def construct(self):
        self.camera.background_color = "#e6d8bc"
        Text.set_default(font="Zalando Sans")
        # this scene starts at 1:39
        line_width = 12
        # add satellite
        sat = MeshReflectorSatSideView().flip(UP).scale(0.5).shift(5 * LEFT + 2 * UP)

        # add water
        water2 = FunctionGraph(
            sine_func, x_range=[-3.2, 3.7], stroke_color=DARK_BLUE, stroke_width=7
        ).shift(2.5 * DOWN)

        golden_gate = SVGMobject(GOLDEN_GATE_PATH, fill_opacity=1)
        golden_gate.set_color(INTERNATIONAL_ORANGE).set_stroke(DARK_RED, width=5)
        golden_gate.scale(2.7).shift(2.5 * RIGHT)

        water1 = FunctionGraph(
            sine_func, x_range=[-5.5, 5.5], stroke_color=DARK_BLUE, stroke_width=12
        ).shift(2.72 * DOWN)

        water1.set_z_index(2)
        water2.set_z_index(11)
        # add bridge, fade out towers and pillars
        bridge1 = (
            Rectangle(
                height=0.195,
                width=1.175,
                color=INTERNATIONAL_ORANGE,
                fill_color=INTERNATIONAL_ORANGE,
                fill_opacity=1,
            )
            .set_stroke(DARK_RED, width=5)
            .shift(2.5 * RIGHT + 1.205 * DOWN)
        )

        bridge2 = (
            Rectangle(
                height=0.3,
                width=3,
                color=INTERNATIONAL_ORANGE,
                fill_color=INTERNATIONAL_ORANGE,
                fill_opacity=1,
            )
            .shift(1.5 * RIGHT + 0.5 * DOWN)
            .set_stroke(DARK_RED, width=5)
        )
        bridge2.set_z_index = 2
        sat.shift(0.7 * UP).scale(0.7)
        water1 = water2
        bridge1 = bridge2
        self.play(FadeIn(water1, bridge1))
        sat.shift(0.5 * RIGHT + UP)
        bridge1.z_index = 3

        direction_db = bridge1.get_left() - sat.get_center()
        start = bridge1.get_left() + 0.1 * DOWN
        traj_sb = [
            sat.get_center() + 0.1 * UP + direction_db * 0.3,
            bridge1.get_left() + 0.1 * UP,
            sat.get_center() + 0.1 * UP + direction_db * 0.3,
        ]
        unit_dir = normalize(direction_db) * np.array([-1, 1, 0])
        for t in np.linspace(0, 20, 100000):
            point = start + t * unit_dir
            if (
                water1.get_left()[0] <= point[0] <= water1.get_right()[0]
                and water1.get_bottom()[1] <= point[1] <= water1.get_top()[1]
            ):
                distance_water_bridge_db = t
                break

        traj_db = [
            sat.get_center() + 0.1 * DOWN + direction_db * 0.3,
            bridge1.get_left() + 0.1 * DOWN,
            bridge1.get_left() + 0.1 * DOWN + unit_dir * distance_water_bridge_db,
            water1.get_left() + 1.2 * RIGHT - direction_db * 0.6,
        ]

        # Cast a ray and find where it hits bridge1
        start = water1.get_left() + 2 * RIGHT
        unit_dir = normalize(direction_db) * np.array([1, -1, 0])
        for t in np.linspace(0, 20, 100000):
            point = start + t * unit_dir
            if (
                bridge1.get_left()[0] <= point[0] <= bridge1.get_right()[0]
                and bridge1.get_bottom()[1] <= point[1] <= bridge1.get_top()[1]
            ):
                distance_water_bridge_tb = t
                break

        traj_tb = [
            water1.get_left() + 2.1 * RIGHT - direction_db * 0.72,
            water1.get_left() + 2 * RIGHT + 0.05 * UP,
            water1.get_left() + 2 * RIGHT + distance_water_bridge_tb * unit_dir,
            water1.get_left() + 2 * RIGHT + 0.05 * UP,
            water1.get_left() + 2.1 * RIGHT - direction_db * 0.72,
        ]

        _, dot_sb, curve_sb = make_beam(traj_sb, COLOR_SB, line_width * 0.5)
        _, dot_db, curve_db = make_beam(traj_db, COLOR_DB, line_width * 0.5)
        _, dot_tb, curve_tb = make_beam(traj_tb, COLOR_TB, line_width * 0.5)

        triggered_sb = False
        sb_finished = False

        def updater_sb(m):
            nonlocal triggered_sb
            nonlocal sb_finished

            if not triggered_sb and np.linalg.norm(m.get_center() - traj_sb[1]) < 0.1:
                m.rotate(PI)
                triggered_sb = True
            if triggered_sb and not sb_finished:
                if np.linalg.norm(m.get_center() - traj_sb[2]) < 0.1:
                    sb_finished = True

        triggered_db_1 = False
        triggered_db_2 = False
        db_finished = False

        def updater_db(m):
            nonlocal triggered_db_1
            nonlocal triggered_db_2
            nonlocal db_finished
            if not triggered_db_1 and np.linalg.norm(m.get_center() - traj_db[1]) < 0.1:
                p0, p1, p2 = traj_db[0], traj_db[1], traj_db[2]
                v1 = p1 - p0
                v2 = p2 - p1
                angle = PI - np.arccos(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                )
                m.rotate(angle + PI)
                triggered_db_1 = True

            if not triggered_db_2 and np.linalg.norm(m.get_center() - traj_db[2]) < 0.1:
                p0, p1, p2 = traj_db[1], traj_db[2], traj_db[3]
                v1 = p1 - p0
                v2 = p2 - p1
                angle = np.arccos(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                )
                m.rotate(angle + PI)
                triggered_db_2 = True

            if triggered_db_2 and triggered_db_1 and not db_finished:
                if np.linalg.norm(m.get_center() - traj_db[3]) < 0.1:
                    db_finished = True
            if db_finished:
                return

        triggered_tb_1 = False
        triggered_tb_2 = False
        triggered_tb_3 = False
        tb_finished = False

        def updater_tb(m):
            nonlocal triggered_tb_1
            nonlocal triggered_tb_2
            nonlocal triggered_tb_3
            nonlocal tb_finished
            if (
                not triggered_tb_1
                and np.linalg.norm(m.get_center() - traj_tb[1]) < 0.05
            ):
                p0, p1, p2 = traj_tb[0], traj_tb[1], traj_tb[2]
                v1 = p1 - p0
                v2 = p2 - p1
                angle = np.arccos(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                )
                m.rotate(angle + 2 * PI)
                triggered_tb_1 = True

            if (
                not triggered_tb_2
                and np.linalg.norm(m.get_center() - traj_tb[2]) < 0.05
            ):
                m.rotate(PI)
                triggered_tb_2 = True

            if (
                not triggered_tb_3
                and triggered_tb_1
                and triggered_tb_2
                and np.linalg.norm(m.get_center() - traj_tb[3]) < 0.05
            ):
                p0, p1, p2 = traj_tb[0], traj_tb[1], traj_tb[2]
                v1 = p1 - p0
                v2 = p2 - p1
                angle = np.arccos(
                    np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                )
                m.rotate(-(angle + 2 * PI))
                triggered_tb_3 = True

            if triggered_tb_1 and triggered_tb_2 and triggered_tb_3 and not tb_finished:
                tb_finished = True

        dot_sb.add_updater(updater_sb)
        dot_sb.set_z_index = 0
        dot_db.add_updater(updater_db)
        dot_db.set_z_index = 0
        dot_tb.add_updater(updater_tb)
        dot_tb.set_z_index = 0

        speed = 4
        trail_sb = TracedPath(
            dot_sb.get_center, stroke_color=COLOR_SB, stroke_width=line_width * 0.5
        )
        trail_db = TracedPath(
            dot_db.get_center, stroke_color=COLOR_DB, stroke_width=line_width * 0.5
        )
        trail_tb = TracedPath(
            dot_tb.get_center, stroke_color=COLOR_TB, stroke_width=line_width * 0.5
        )

        trail_sb.z_index = 0
        trail_db.z_index = 0
        trail_tb.z_index = 1
        self.camera.frame.save_state()
        self.add(trail_sb, trail_db, trail_tb)

        triggered_sb = False
        triggered_db_1 = False
        triggered_db_2 = False
        triggered_tb_1 = False
        triggered_tb_2 = False
        triggered_tb_3 = False
        sb_finished = False
        db_finished = False
        tb_finished = False

        self.add(trail_sb, trail_db, trail_tb)

        self.play(
            MoveAlongPath(
                dot_sb,
                curve_sb,
                run_time=curve_sb.get_arc_length() / speed,
                rate_func=linear,
            ),
            MoveAlongPath(
                dot_db,
                curve_db,
                run_time=curve_db.get_arc_length() / speed,
                rate_func=linear,
            ),
            MoveAlongPath(
                dot_tb,
                curve_tb,
                run_time=curve_tb.get_arc_length() / (speed * 1.1),
                rate_func=linear,
            ),
        )

        self.remove(trail_db)
        curve_sb.z_index = 0
        curve_db.z_index = 0
        curve_tb.z_index = 1
        self.add(curve_sb, curve_db, curve_tb)

        # hard cut
        self.play(
            FadeOut(dot_tb, curve_tb, trail_tb), run_time=0.5
        )  # don't need triple bounce from now on

        # draw clearance of bridge
        line = Line(start=2 * UP, end=(bridge1.get_left() + 0.195 / 2 * UP), color=GREY)
        bridge_height = DashedLine(
            start=bridge1.get_left()
            + (bridge1.get_left()[1] - water1.get_center()[1]) * DOWN,
            end=bridge1.get_left() + 0.195 / 2 * DOWN,
            color=BLACK,
        )
        bridge_height_label_old = (
            MathTex(r"\textsf{h}", color=BLACK)
            .set_stroke(BLACK)
            .next_to(bridge_height, RIGHT)
        )
        bridge_height_label = (
            MathTex(r"\textsf{h}", color=BLACK)
            .set_stroke(BLACK)
            .next_to(bridge_height, RIGHT)
        )
        bridge_height_label2 = (
            Text(
                r"clearance",
                color=BLACK,
                font="Zalando Sans",
                font_size=SOURCE_FONT_SIZE,
            )
            .scale(SOURCE_SCALE * 2)
            .set_stroke(BLACK)
            .move_to(bridge_height_label_old.get_center())
            .align_to(bridge_height_label_old, LEFT)
        )
        self.play(Create(bridge_height))
        self.play(FadeIn(bridge_height_label_old))
        self.wait(1)
        self.play(Transform(bridge_height_label_old, bridge_height_label2))
        self.wait(1)
        self.play(ReplacementTransform(bridge_height_label_old, bridge_height_label))
        self.wait(1)

        scale_factor_dot = 2
        self.play(
            curve_sb.animate.set_stroke(width=line_width),
            dot_sb.animate.scale(scale_factor_dot),
            run_time=0.5,
        )
        self.play(
            curve_sb.animate.set_stroke(width=line_width * 0.5),
            dot_sb.animate.scale(1 / scale_factor_dot),
            run_time=0.5,
        )
        self.play(
            curve_db.animate.set_stroke(width=line_width),
            dot_db.animate.scale(scale_factor_dot),
            run_time=0.5,
        )
        self.play(
            curve_db.animate.set_stroke(width=line_width * 0.5),
            dot_db.animate.scale(1 / scale_factor_dot),
            run_time=0.5,
        )
        self.wait(1)

        # draw incidence angle
        # we need to shift the line a bit down to draw the angle,
        # otherwise the angle is on the bridge
        line_shift = Line(
            start=2 * UP, end=(bridge1.get_left() + 0.195 / 2 * DOWN), color=GREY
        )
        line_db = Line(start=traj_db[0], end=traj_db[1])
        line_db_invert = Line(start=traj_db[1], end=traj_db[0])
        top_angle = draw_angle(
            line_shift, line_db, radius=1.2, color=GREY, opposite_angle=False
        ).shift(0.195 * UP)
        filled_top_angle = Angle(
            line_db_invert,
            line_shift,
            radius=1.2,
            color=BLACK,
            quadrant=(1, -1),
            other_angle=True,
        ).shift(0.195 * UP)
        angle_label = (
            MathTex(r"\mathsf{\theta}", color=BLACK)
            .set_stroke(BLACK)
            .move_to(filled_top_angle.get_center() + 0.25 * DOWN + 0.17 * RIGHT)
        )
        self.play(Create(line))
        self.play(FadeIn(top_angle, filled_top_angle))
        self.play(FadeIn(angle_label))
        incidence_text = (
            Text(
                "incidence angle",
                font="Zalando Sans",
                font_size=SOURCE_FONT_SIZE,
                color=BLACK,
                stroke_color=BLACK,
            )
            .scale(SOURCE_SCALE * 2)
            .next_to(bridge1, UP)
            .shift(0.1 * RIGHT)
        )
        self.play(FadeIn(incidence_text))
        self.wait(2)

        sat2 = MeshReflectorSatSideView().flip(UP).scale(0.5).shift(8 * LEFT + 5 * UP)
        sat2.z_index = 2
        self.add(sat2)
        sat_pos_label = (
            MathTex(r"\textsf{(x,y,z)}", color=BLACK, font_size=SOURCE_FONT_SIZE * 1.5)
            .next_to(sat2, RIGHT)
            .set_stroke(BLACK)
        )
        self.play(self.camera.frame.animate.scale(1.6).shift(2 * UP))
        self.play(Write(sat_pos_label))
        self.wait(0.5)
        self.play(FadeOut(sat_pos_label))

        self.wait(1)

        cloud1 = (
            Cloud(cloud_type=1)
            .shift(7 * UP + 5 * RIGHT)
            .scale(1)
            .set_fill(opacity=0.5, color=GREY_A)
        )
        cloud2 = (
            Cloud(cloud_type=2)
            .shift(6 * UP + 15 * LEFT)
            .scale(1.2)
            .set_fill(opacity=0.5, color=GREY_A)
        )
        cloud3 = (
            Cloud(cloud_type=1)
            .shift(6 * UP + 1.5 * RIGHT)
            .scale(0.9)
            .set_fill(opacity=0.5, color=GREY_A)
        )
        cloud4 = (
            Cloud(cloud_type=2)
            .shift(7 * UP + 10 * RIGHT)
            .scale(0.7)
            .set_fill(opacity=0.5, color=GREY_A)
        )
        cloud5 = (
            Cloud(cloud_type=1)
            .shift(5 * UP + 8 * LEFT)
            .set_fill(opacity=0.5, color=GREY_A)
        )
        clouds = [cloud1, cloud2, cloud3, cloud4, cloud5]
        self.play(
            sat2.animate.shift(12 * LEFT + 9 * UP),
            self.camera.frame.animate.scale(2).shift(4 * UP + 2 * LEFT),
            FadeIn(*clouds),
            run_time=1.5,
        )
        dist_line = DashedLine(
            start=sat2.get_center(),
            end=bridge_height.get_bottom(),
            color=BLACK,
            dash_length=0.5,
        ).set_stroke(width=8)
        dist_line.z_index = 0
        dist_label = (
            Text(">500km", color=BLACK, font_size=SOURCE_FONT_SIZE * 2)
            .next_to(dist_line, RIGHT, buff=0.2)
            .shift(11 * LEFT + 2.5 * UP)
        )
        self.play(Create(dist_line))
        self.play(FadeIn(dist_label))
        self.play(FadeOut(dist_line, *clouds, dist_label))
        self.play(
            self.camera.frame.animate.scale(1 / (1.6 * 2)).shift(6 * DOWN + 2 * RIGHT)
        )
        self.play(FadeOut(incidence_text))

        descr_text = (
            Text("Description", font_size=SOURCE_FONT_SIZE, color=BLACK)
            .scale(SOURCE_SCALE * 2)
            .move_to(water1.get_bottom() + 1 * DOWN)
        )
        descr_arrow = Arrow(
            start=descr_text.get_bottom() + 0.1 * DOWN,
            end=descr_text.get_bottom() + 0.5 * DOWN,
            color=BLACK,
        ).set_stroke(width=4)

        self.play(
            FadeIn(descr_text),
            GrowArrow(descr_arrow),
            self.camera.frame.animate.shift(DOWN),
        )
        self.wait(1)
        self.play(FadeOut(descr_text, descr_arrow), self.camera.frame.animate.shift(UP))

        # draw single bounce with double bounce length
        # should be parallel to single bounce
        direction_sb = traj_sb[0] - traj_sb[1]
        direction_sb = direction_sb / np.linalg.norm(direction_sb)
        start_point = (
            bridge1.get_left() + (bridge1.get_left()[1] - water1.get_center()[1]) * DOWN
        )
        distance = 5
        traj_fake_db = [
            start_point + direction_sb * distance,
            start_point,
            start_point + direction_sb * distance,
        ]
        _, dot_fake_db, curve_fake_db = make_beam(
            traj_fake_db, COLOR_FAKE_DB, line_width * 0.5
        )

        triggered_fake_db = False
        fake_db_finished = False

        def updater_fake_db(m):
            nonlocal triggered_fake_db
            nonlocal fake_db_finished

            if (
                not triggered_fake_db
                and np.linalg.norm(m.get_center() - traj_fake_db[1]) < 0.1
            ):
                m.rotate(PI)
                triggered_fake_db = True
            if triggered_fake_db and not fake_db_finished:
                if np.linalg.norm(m.get_center() - traj_fake_db[2]) < 0.1:
                    fake_db_finished = True

        dot_fake_db.add_updater(updater_fake_db)

        trail_fake_db = TracedPath(
            dot_fake_db.get_center,
            stroke_color=COLOR_FAKE_DB,
            stroke_width=line_width * 0.5,
        )
        trail_fake_db.z_index = 0

        virt_text = (
            Text("virtual scatterer", font_size=SOURCE_FONT_SIZE, color=COLOR_FAKE_DB)
            .scale(SOURCE_SCALE * 2.3)
            .shift(3 * UP)
        )
        source_text = (
            Text(
                "Ludovic Villard, P. Borderies. On the use of virtual ground scatterers to localize double and triple bounce scattering mechanisms\n"
                "for bistatic SAR. Journal of Electromagnetic Waves and Applications, 2015, 29 (5), pp.626-635",
                font_size=SOURCE_FONT_SIZE,
                color=SOURCE_COLOR,
            )
            .scale(SOURCE_SCALE)
            .to_corner(DL)
        )
        self.play(FadeIn(virt_text, source_text))

        self.add(trail_fake_db)
        self.play(
            MoveAlongPath(
                dot_fake_db,
                curve_fake_db,
                run_time=curve_fake_db.get_arc_length() / speed,
                rate_func=linear,
            )
        )
        curve_fake_db.z_index = 0
        self.add(curve_fake_db)

        line_horizontal = Line(
            start=water1.get_right(), end=(water1.get_left()), color=GREY
        )
        line_db = Line(start=traj_db[1], end=traj_db[2])
        line_db_invert = Line(start=traj_db[3], end=traj_db[2])
        top_angle1 = draw_angle(
            line_db, line_horizontal, radius=1.2, color=GOLD_C, opposite_angle=False
        ).set_stroke(width=3, color=GOLD_E)
        top_angle2 = (
            draw_angle(
                line_db, line_horizontal, radius=1.2, color=GOLD_C, opposite_angle=False
            )
            .set_stroke(width=3, color=GOLD_E)
            .flip()
            .shift(LEFT * 1.2)
        )
        top_angle3 = draw_angle(
            line_db_invert, line_db, radius=1.2, color=GOLD_C, opposite_angle=False
        ).set_stroke(width=3, color=GOLD_E)
        self.play(FadeIn(top_angle1))
        self.play(FadeOut(top_angle1), FadeIn(top_angle2))
        self.play(FadeOut(top_angle2), FadeIn(top_angle3))
        self.play(FadeOut(top_angle3))

        self.play(FadeOut(dot_db, curve_db))

        # draw incidence angle again
        bridge_height_invert = DashedLine(
            start=bridge1.get_left() + 0.195 / 2 * DOWN,
            end=bridge1.get_left()
            + (bridge1.get_left()[1] - water1.get_center()[1]) * DOWN,
            color=BLACK,
        )
        line_fake_db = Line(start=traj_fake_db[0], end=traj_fake_db[1], color=GREY)
        top_angle_fake_db = draw_angle(
            bridge_height_invert,
            line_fake_db,
            radius=1.2,
            color=GREY,
            opposite_angle=False,
        )
        filled_top_angle_fake_db = Angle(
            line_fake_db,
            bridge_height,
            radius=1.2,
            color=BLACK,
            quadrant=(-1, 1),
            other_angle=True,
        )  # .shift(.195*UP)
        self.play(FadeIn(top_angle_fake_db, filled_top_angle_fake_db))
        angle_label_fake_db = (
            MathTex(r"\mathsf{\theta}", color=BLACK)
            .set_stroke(BLACK)
            .move_to(filled_top_angle_fake_db.get_center() + 0.25 * DOWN + 0.18 * RIGHT)
        )
        angle_label_fake_db_t = (
            MathTex(r"\mathsf{\theta}", color=BLACK)
            .set_stroke(BLACK)
            .move_to(filled_top_angle_fake_db.get_center() + 0.5 * DOWN + 0.23 * RIGHT)
            .scale(0.75)
        )
        self.play(FadeIn(angle_label_fake_db))
        self.play(FadeOut(filled_top_angle, top_angle, angle_label))

        top_angle_fake_db_t = draw_angle(
            bridge_height_invert,
            line_fake_db,
            radius=0.8,
            color=GREY,
            opposite_angle=False,
        )
        filled_top_angle_fake_db_t = Angle(
            line_fake_db,
            bridge_height,
            radius=0.8,
            color=BLACK,
            quadrant=(-1, 1),
            other_angle=True,
        )
        self.play(
            Transform(top_angle_fake_db, top_angle_fake_db_t),
            Transform(filled_top_angle_fake_db, filled_top_angle_fake_db_t),
            Transform(angle_label_fake_db, angle_label_fake_db_t),
            FadeOut(source_text, virt_text),
        )

        # draw right-angle
        # Direction of traj_fake_db
        direction = (traj_fake_db[1] - traj_fake_db[0]) / norm(
            traj_fake_db[1] - traj_fake_db[0]
        )

        # Perpendicular direction (rotate 90°)
        perpendicular = np.array([-direction[1], direction[0], 0])

        # Find intersection using parametric equations
        # traj_fake_db[0] + t * direction = bridge.get_left() + s * perpendicular
        p = traj_fake_db[0][:2]
        d = direction[:2]
        q = bridge1.get_left()[:2]
        n = perpendicular[:2]

        t = ((q - p)[0] * n[1] - (q - p)[1] * n[0]) / (d[0] * n[1] - d[1] * n[0])

        intersection = p + t * d
        intersection = np.array([intersection[0], intersection[1], 0])

        # Draw line from bridge to intersection
        perpendicular_line = Line(
            bridge1.get_left(), intersection, color=GREY, stroke_width=line_width * 0.5
        )
        # Draw right angle with dot
        right_angle = RightAngle(
            Line(intersection, intersection + direction),
            Line(intersection, intersection + perpendicular),
            length=0.2,
            color=GREY,
            dot=True,
            dot_color=BLACK,
        )
        self.play(self.camera.frame.animate.scale(0.8).shift(0.5 * DOWN))
        self.play(Create(perpendicular_line))
        self.play(Create(right_angle))
        self.wait(1)

        # Highlight triangle: vertices are traj_fake_db[0], intersection, bridge.get_left()
        triangle = VMobject()

        triangle.set_points_as_corners(
            [
                traj_fake_db[1],
                intersection,
                bridge1.get_left(),
                traj_fake_db[1],
            ]
        )

        triangle.set_fill(WHITE, opacity=0.3)
        triangle.set_stroke(GREY, width=line_width * 0.5)
        for (line,) in triangle:
            line.joint_type = manim.constants.LineJointType.BEVEL

        triangle.set_z_index(2)

        bridge_height.z_index = 3
        self.add(bridge_height)
        triangle.z_index = 2
        self.play(FadeIn(triangle))
        self.wait(1)

        # draw R_db - R_sb
        diff_sb_db = Line(
            start=traj_fake_db[1],
            end=intersection,
            color=RED_D,
            stroke_width=line_width * 0.5,
        )
        diff_sb_db.z_index = 3
        water1.z_index = 4
        bridge1.z_index = 4
        self.add(water1, bridge1)
        self.play(FadeIn(diff_sb_db))
        text_diff_sb_db = MathTex(
            r"\textsf{R}_{\textsf{db}} - \textsf{R}_{\textsf{sb}}}", color=RED_D
        ).next_to(diff_sb_db, LEFT, buff=0.2)
        self.play(FadeIn(text_diff_sb_db))
        bridge_height = Line(
            start=bridge1.get_left()
            + (bridge1.get_left()[1] - water1.get_center()[1]) * DOWN,
            end=bridge1.get_left() + 0.195 / 2 * DOWN,
            color=SOURCE_COLOR_DARK,
        )
        bridge_height.set_z_index(5)
        self.play(
            FadeIn(bridge_height),
            bridge_height_label.animate.set_color(SOURCE_COLOR_DARK),
        )

        # zoom out and give equation
        self.play(self.camera.frame.animate.shift(3.25 * RIGHT + 0.3 * UP).scale(1.4))

        lhs = MathTex(r"\mathsf{cos}(\mathsf{\theta}) =", color=BLACK).scale(1.6)
        frac = (
            MathTex(
                r"\textsf{adjacent}",
                "\over",
                r"\textsf{hypotenuse}",
            )
            .set_color(BLACK)
            .shift(7 * RIGHT)
            .scale(1.6)
        )
        frac[0].set_color(RED_D)
        frac[-1].set_color(SOURCE_COLOR_DARK)
        expr = VGroup(lhs, frac).arrange(RIGHT, buff=0.2).shift(7 * RIGHT + 0.5 * DOWN)
        self.play(FadeIn(expr))
        self.wait(1)

        lhs1 = MathTex(r"\mathsf{cos}(\mathsf{\theta}) =", color=BLACK).scale(1.6)
        frac2 = (
            MathTex(
                r"\textsf{R}_{\textsf{db}}- \textsf{R}_{\textsf{sb}}",
                "\over",
                r"\textsf{hypotenuse}",
            )
            .set_color(BLACK)
            .shift(7 * RIGHT)
            .scale(1.6)
        )
        frac2[0].set_color(RED_D)
        frac2[-1].set_color(SOURCE_COLOR_DARK)
        expr1 = (
            VGroup(lhs1, frac2).arrange(RIGHT, buff=0.2).shift(7 * RIGHT + 0.5 * DOWN)
        )

        self.play(Transform(expr, expr1))
        self.wait(1)

        lhs2 = MathTex(r"\mathsf{cos}(\mathsf{\theta}) =", color=BLACK).scale(1.6)
        frac3 = (
            MathTex(
                r"\textsf{R}_{\textsf{db}}- \textsf{R}_{\textsf{sb}}",
                "\over",
                r"\textsf{h}",
            )
            .set_color(BLACK)
            .shift(7 * RIGHT)
            .scale(1.6)
        )
        frac3[0].set_color(RED_D)
        frac3[-1].set_color(SOURCE_COLOR_DARK)
        expr2 = (
            VGroup(lhs2, frac3).arrange(RIGHT, buff=0.2).shift(7 * RIGHT + 0.5 * DOWN)
        )
        self.play(Transform(expr, expr2))
        self.wait(1)

        lhs1 = MathTex(r"\mathsf{h} =", color=BLACK).scale(1.6)
        frac3 = (
            MathTex(
                r"\textsf{R}_{\textsf{db}}- \textsf{R}_{\textsf{sb}}",
                "\over",
                r"\mathsf{cos}(\mathsf{\theta})}",
            )
            .set_color(BLACK)
            .shift(7 * RIGHT)
            .scale(1.6)
        )
        frac3[0].set_color(RED_D)
        lhs1[0].set_color(SOURCE_COLOR_DARK)
        expr3 = (
            VGroup(lhs1, frac3).arrange(RIGHT, buff=0.2).shift(7 * RIGHT + 0.5 * DOWN)
        )
        self.play(Transform(expr, expr3))
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)
