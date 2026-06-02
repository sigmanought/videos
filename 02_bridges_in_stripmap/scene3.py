"Illustration of the coordinate system of a SAR image." ""
from manim import *
from utils.colors import *
from utils.decoration import create_shadow
from utils.objects import Clock
from utils.sat import MeshReflectorSat

fast = 1
FILES_DIR = "./02_bridges_in_stripmap/pngs/"
SAR_IMAGE_PATH = f"{FILES_DIR}/placeholder.png"


class Scene3(MovingCameraScene):
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

        self.play(FadeIn(sar_shadow, sar_image, sar_source), run_time=fast * 1)
        self.play(sar_shadow.animate.scale(0.8), sar_image.animate.scale(0.8))

        arrow_origin = sar_image.get_corner(UP + LEFT) + (0.7 * UP + 0.7 * LEFT)
        down_factor = 4
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

        text_range = (
            Text(
                "Longitude",
                font="Zalando Sans",
                font_size=SOURCE_FONT_SIZE,
                color=BLACK,
            )
            .move_to(arrow_up.get_end() + 0.5 * DOWN)
            .scale(SOURCE_SCALE * 2)
        )
        text_azimuth = (
            Text(
                "Latitude", font="Zalando Sans", font_size=SOURCE_FONT_SIZE, color=BLACK
            )
            .move_to(arrow_right.get_end() + 1 * RIGHT)
            .scale(SOURCE_SCALE * 2)
        )

        self.play(
            FadeIn(arrow_up, arrow_right),
            FadeIn(text_range),
            FadeIn(text_azimuth),
            run_time=fast * 1,
        )
        self.wait(1)

        # Create cross-out lines
        cross_range1 = Line(
            text_range.get_corner(DL),
            text_range.get_corner(UR),
            color=RED,
            stroke_width=3,
        )
        cross_range2 = Line(
            text_range.get_corner(UL),
            text_range.get_corner(DR),
            color=RED,
            stroke_width=3,
        )

        cross_azimuth1 = Line(
            text_azimuth.get_corner(DL),
            text_azimuth.get_corner(UR),
            color=RED,
            stroke_width=3,
        )

        cross_azimuth2 = Line(
            text_azimuth.get_corner(UL),
            text_azimuth.get_corner(DR),
            color=RED,
            stroke_width=3,
        )

        # Animate the cross-outs
        cross1 = VGroup(cross_range1, cross_range2)
        cross2 = VGroup(cross_azimuth1, cross_azimuth2)
        self.play(FadeIn(cross1), FadeIn(cross2), run_time=fast * 0.75)
        self.wait(fast * 1)

        sat = (
            MeshReflectorSat(radius=2, num_scallops=12, scallop_depth=0.7)
            .set_stroke(1)
            .scale(0.2)
            .rotate(PI)
        )

        rounded_rect = (
            RoundedRectangle(corner_radius=0.15, height=0.35, width=sar_image.width)
            .set_fill([RED, BLUE], opacity=1)
            .set_stroke(BLACK, width=4)
        )
        rounded_rect.next_to(sar_image, UP, buff=1)
        sat.next_to(rounded_rect, LEFT, buff=0)

        self.play(
            FadeOut(cross1, cross2, text_range, text_azimuth, arrow_right, arrow_up)
        )
        self.play(FadeIn(sat))
        self.play(
            sat.animate.move_to(rounded_rect.get_right()), run_time=3, rate_func=linear
        )

        new_text_range = (
            Text("Range", font_size=SOURCE_FONT_SIZE, font="Zalando Sans", color=BLACK)
            .move_to(text_range.get_center())
            .scale(SOURCE_SCALE * 2)
        )

        new_text_azimuth = (
            Text(
                "Azimuth", font_size=SOURCE_FONT_SIZE, font="Zalando Sans", color=BLACK
            )
            .move_to(text_azimuth.get_center())
            .scale(SOURCE_SCALE * 2)
        )

        # Transform animations
        self.play(FadeIn(arrow_right), FadeOut(sat), run_time=fast * 1)
        self.play(FadeIn(new_text_azimuth), run_time=fast * 1)
        self.wait(2)
        self.play(FadeIn(arrow_up), run_time=fast * 1)
        self.play(FadeIn(new_text_range), run_time=fast * 1)
        self.wait(fast * 1)

        self.wait(fast * 2)

        self.play(
            FadeOut(new_text_range, new_text_azimuth, arrow_up, arrow_right),
            run_time=fast * 1,
        )

        self.play(
            sar_shadow.animate.scale(1 / 0.8),
            sar_image.animate.scale(1 / 0.8),
            run_time=fast * 1,
        )
        self.wait(3)
        self.play(FadeIn(grid))
        self.wait(1)

        clock = Clock(radius=1, stroke_width=6).scale(0.6)
        clock.move_to(sar_image.get_top() + 0.8 * UP)

        self.play(
            sar_image.animate.shift(0.5 * DOWN),
            sar_shadow.animate.shift(0.5 * DOWN),
            grid.animate.shift(0.5 * DOWN),
            FadeIn(clock),
        )
        self.wait(1)
        self.play(FadeOut(clock))
        self.play(
            sar_image.animate.shift(0.5 * UP),
            sar_shadow.animate.shift(0.5 * UP),
            grid.animate.shift(0.5 * UP),
        )

        left_factor = 1.5
        arrow_origin = sar_image.get_corner(UP + LEFT) + (0.7 * UP)
        arrow_right = Arrow(
            start=arrow_origin,
            end=arrow_origin + RIGHT * left_factor,
            stroke_color=BLACK,
            stroke_width=3,
            tip_length=0.2,
            buff=0,
        ).shift(0.25 * UP)
        text_azimuth = (
            Text(
                "Azimuth", font_size=SOURCE_FONT_SIZE, font="Zalando Sans", color=BLACK
            )
            .move_to(arrow_right.get_end() + 1 * RIGHT)
            .scale(SOURCE_SCALE * 2)
        )
        self.play(FadeIn(arrow_right, text_azimuth))
        self.wait(1)

        rounded_rect = (
            RoundedRectangle(corner_radius=0.15, height=0.35, width=sar_image.width)
            .set_fill([RED, BLUE], opacity=1)
            .set_stroke(BLACK, width=4)
        )
        rounded_rect.next_to(sar_image, UP, buff=0.5)

        def update_reveal(mob, alpha):
            # alpha goes from 0 to 1 over the animation
            current_width = sar_image.width * alpha

            if current_width > 0:
                new_rect = RoundedRectangle(
                    corner_radius=0.15, height=0.35, width=current_width
                ).set_stroke(BLACK, width=4)

                # Left edge is always RED (progress = 0)
                # Right edge progresses from RED (alpha=0) to BLUE (alpha=1)
                start_color = RED  # Left edge
                end_color = interpolate_color(BLUE, RED, 1 - alpha)

                new_rect.set_fill([end_color, start_color], opacity=1)
                new_rect.next_to(sar_image, UP, buff=0.5)
                new_rect.align_to(rounded_rect, LEFT)  # Align to the right edge

                mob.become(new_rect)
                mob.z_index = 0

        # Set initial state
        rounded_rect.set_opacity(0)

        sat = (
            MeshReflectorSat(radius=2, num_scallops=12, scallop_depth=0.7)
            .set_stroke(1)
            .scale(0.2)
            .rotate(PI)
        )
        sat.next_to(rounded_rect, LEFT, buff=0)
        sat.z_index = 1

        time_labels = VGroup()
        times = [0, 1, 2, 3, 4]  # seconds
        shown = [False, False, False, False, False]

        for i, time in enumerate(times):
            label = Text(
                f"{time}sec",
                font_size=24,
                font="Zalando Sans",
            ).set_color(BLACK)
            # Position at quarters of the rectangle width
            x_position = (
                rounded_rect.get_left()[0] + (i / (len(times) - 1)) * sar_image.width
            )
            label.move_to([x_position, rounded_rect.get_top()[1] + 0.3, 0])
            label.z_index = 0
            time_labels.add(label)

        flying_speed_fac = 3  # high values are slower

        def show_labels_updater(mob, alpha):
            elapsed = alpha * 4 * flying_speed_fac  # 4 second animation
            if elapsed >= 0 and not shown[0]:
                time_labels[0].z_index = 0
                self.add(time_labels[0])
                self.bring_to_front(sat)
                shown[0] = True
            if elapsed >= 1 * flying_speed_fac and not shown[1]:
                time_labels[1].z_index = 0
                self.add(time_labels[1])
                self.bring_to_front(sat)
                shown[1] = True
            if elapsed >= 2 * flying_speed_fac and not shown[2]:
                time_labels[2].z_index = 0
                self.add(time_labels[2])
                self.bring_to_front(sat)
                shown[2] = True
            if elapsed >= 3 * flying_speed_fac and not shown[3]:
                time_labels[3].z_index = 0
                self.add(time_labels[3])
                self.bring_to_front(sat)
                shown[3] = True
            if elapsed >= 4 * flying_speed_fac and not shown[4]:
                time_labels[4].z_index = 0
                self.add(time_labels[4])
                self.bring_to_front(sat)
                shown[4] = True

        # Create the timer
        timer = DecimalNumber(
            0, num_decimal_places=4, font_size=SOURCE_FONT_SIZE
        ).scale(SOURCE_SCALE * 3)
        clock = Clock(radius=1, stroke_width=3).scale(0.3)
        timer_group = (
            VGroup(
                clock,
                VGroup(
                    Text("t = ", font_size=SOURCE_FONT_SIZE, font="Zalando Sans").scale(
                        SOURCE_SCALE * 2.5
                    ),
                    timer,
                    Text("sec", font_size=SOURCE_FONT_SIZE, font="Zalando Sans").scale(
                        SOURCE_SCALE * 2.5
                    ),
                ).arrange(RIGHT, buff=0.1, aligned_edge=DOWN),
            )
            .arrange(RIGHT, buff=0.3)
            .set_color(BLACK)
        )
        timer_group.next_to(sar_image, DOWN, buff=0.75)

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

        for col_index in [3, 21, 30, 12, 17]:  # any indices you want

            start = col_index * num_rows
            end = start + num_rows

            column = boxes[start:end]
            column_copy = [box.copy() for box in column]

            for box in column_copy:
                box.set_fill(opacity=0.8).set_color(RED).shift(0.01 * RIGHT)

            self.play(FadeIn(*column_copy), run_time=0.3)
            self.wait(0.5)
            self.play(FadeOut(*column_copy), run_time=0.3)
        self.play(FadeOut(arrow_right, text_azimuth))

        sat.z_index = 1
        self.play(FadeIn(sat))
        self.play(FadeIn(timer_group))

        # Animate filling boxes sequentially
        self.play(
            UpdateFromAlphaFunc(rounded_rect, update_reveal),
            UpdateFromAlphaFunc(VGroup(), show_labels_updater),
            sat.animate.move_to(rounded_rect.get_right()),
            ChangeDecimalToValue(timer, 4),
            LaggedStart(*[FadeIn(box, run_time=0.2) for box in boxes], lag_ratio=0.2),
            run_time=4 * flying_speed_fac,
            rate_func=linear,
        )

        self.play(FadeOut(sat))
        self.wait(4)
        self.play(FadeOut(*boxes, timer_group, *time_labels, rounded_rect))

        col_index = 20
        start = col_index * num_rows
        end = start + num_rows

        column = boxes[start:end]
        column_copy = [box.copy() for box in column]

        for box in column_copy:
            box.set_fill(opacity=0.8).set_color(RED).shift(0.01 * RIGHT)

        arrow_up.shift(1 * LEFT)
        new_text_range.shift(0.5 * LEFT)
        self.wait(3)
        self.play(FadeIn(arrow_up), run_time=1)
        self.play(FadeIn(new_text_range), run_time=1)
        self.wait(1)
        self.play(
            FadeIn(*column_copy),
        )
        self.wait(1)

        self.play(
            FadeOut(
                sar_image,
                sar_source,
                grid,
                sar_shadow,
                *column_copy,
                new_text_range,
                arrow_up,
            ),
            run_time=fast * 1,
        )
