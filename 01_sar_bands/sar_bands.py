"""Explanation of different bands in SAR."""
from manim import *  # not good practice, but handy
import numpy as np
import random
from objects import (
    House,
    Car,
    Tree,
    Cloud,
    Raindrop,
    Sun,
    Moon,
    Sat,
    Dune,
    Cave,
    Crosshair,
    SecretDocument,
)


def create_glow(
    vmobject,
    layers=15,
    start_stroke=4,
    scale_factor=1.03,
    color=WHITE,
    max_opacity=0.25,
):
    glow_group = VGroup()
    for i in range(layers):
        opacity = max_opacity * (1 - i / layers)
        scaled = (
            vmobject.copy()
            .set_stroke(
                width=start_stroke * scale_factor * (i / layers),
                color=color,
                opacity=opacity,
            )
            .set_fill(opacity=0)
        )
        glow_group.add(scaled)
    return glow_group


MathTex.set_default(color=BLACK)

LIGHT_BEIGE = ManimColor("#e8dfcc")
WASHED_BROWN = ManimColor("#746458")
SAT_MOVE_RUNTIME = 0.2
BEAM_MOVE_RUNTIME = 0.6
BEAM_MOVE_RUNTIME_FAST = 0.6
AMPLITUDE_DFLT = 0.2
FREQUENCY_X_BAND = 1 / 0.5
FREQUENCY_K_BAND = 1 / 0.2
FREQUENCY_DFLT = 1 / 0.6
TIME_BETWEEN_BEAMS = 0.05
FADEOUT_TIME = 0.5
SCALE_SAT = 0.8
SAT_HEIGHT = 1.85
GROUND_LINE_POS = -2.5
GROUND_LINE_INTERVAL = [-5, 5]
TITLE_POS = 2.7
TITLE_POS_WAVELENGTH = 0.6
TITLE_POS_BAND_NAME = 1.5
FADE_OUT_COLOR = ManimColor("#746458")
FADE_OUT_OPACITY = 0.5
SOURCE_COLOR = ManimColor("#665950")  # GRAY_BROWN

BORDER_COLOR_BOX = ManimColor("#332b23")

IMG_WIDTH = 10
IMG_HEIGHT = 6


class SARBands(Scene):

    def add_scale(self):
        """Shows a centimeter scale"""
        units_per_cm = 2  # adjust to match your scene scaling
        ruler_length_cm = 4

        # Create ruler line
        ruler = Line(start=LEFT, end=RIGHT, color=BLACK)
        ruler.scale(ruler_length_cm * units_per_cm / 2)
        ruler.move_to(4.75 * DOWN)
        self.add(ruler)

        # Add tick marks and labels
        tick_values_cm = [0, 1, 2, 3, 4]
        ticks = []
        labels = []
        for cm in tick_values_cm:
            # Tick position relative to left end of ruler
            x_pos = ruler.get_left()[0] + cm * units_per_cm
            y_pos = ruler.get_y()  # same y as ruler
            tick = Line(
                start=[x_pos, y_pos - 0.05, 0],  # small line below
                end=[x_pos, y_pos + 0.05, 0],  # small line above
                color=BLACK,
            )
            label = (
                Text(f"{cm} cm", font_size=24)
                .next_to(tick, DOWN, buff=0.1)
                .set_color(BLACK)
            )
            ticks.append(tick)
            labels.append(label)

        tick_group = VGroup(*ticks, *labels)
        self.add(tick_group)
        return ruler, tick_group

    def add_wave(
        self,
        t_tracker,
        position,
        frequency,
        amplitude=0.1,
        x_range=[-1.5, 1.5],
        direction="forward",
        pos_tracker=None,
    ):
        """Add a sine wave at 'position' with wavelength
        1/frequency."""
        if direction == "forward":
            direction = 1
        elif direction == "backward":
            direction = -1

        def sine_wave(x):
            return amplitude * np.sin(
                2 * np.pi * frequency *
                (x + direction * 0.5 * t_tracker.get_value())
            )

        if pos_tracker:
            wave = always_redraw(
                lambda: FunctionGraph(sine_wave, x_range=x_range, color=BLACK)
                .rotate(-90 * DEGREES)
                .move_to(pos_tracker.get_value())
            )
        else:
            wave = always_redraw(
                lambda: FunctionGraph(sine_wave, x_range=x_range, color=BLACK)
                .rotate(-90 * DEGREES)
                .move_to(position)
            )

        self.add(wave)

        return wave

    def create_moving_ground(self, base_points, axes, t_val, flat_directions):
        new_points = []
        flat_idx = 0  # index for flat_directions

        for idx, (x, y) in enumerate(base_points):
            if idx % 2 == 0:  # flat segment
                direction = flat_directions[flat_idx]
                y += direction * 0.5 * t_val
                flat_idx += 1
                new_points.append(axes.c2p(x, y))
            else:  # vertical step: connect previous flat
                y_prev_scene = new_points[-1][1]
                new_points.append(axes.c2p(x, y_prev_scene))

        vm = VMobject(color=BLACK, stroke_width=5)
        vm.set_points_as_corners(new_points)
        return vm

    def play_scene_1(self, fw, fh):

        SAT_HEIGHT_SCENE_1 = SAT_HEIGHT - 0.2
        axes = Axes(
            x_range=[-fw / 2, fw / 2, 1],
            y_range=[-fh / 2, fh / 2, 1],
            x_length=fw,
            y_length=fh,
            tips=False,
        ).set_color(GREY)
        axes.add_coordinates(color=GREY)

        line_pos = GROUND_LINE_POS
        line = Line(  # ground surface
            axes.c2p(GROUND_LINE_INTERVAL[0], line_pos),
            axes.c2p(GROUND_LINE_INTERVAL[1], line_pos),
            color=BLACK,
        )
        self.add(line)

        # trees at left and right
        tree1 = Tree(tree_type=1).scale(
            0.6).align_to(line, DOWN).shift(3 * LEFT)
        tree2 = Tree(tree_type=2).scale(0.4).align_to(
            line, DOWN).shift(2.5 * RIGHT)
        self.add(tree1, tree2)

        # Satellite
        sat = Sat().scale(SCALE_SAT)
        # Lighting and weather conditions
        sun = Sun().scale(0.6).move_to(1.5 * UP + 2.6 * LEFT)
        moon = Moon().scale(0.45).move_to(1.5 * UP + 2.6 * LEFT).rotate(30 * DEGREES)
        cloud1 = (
            Cloud(cloud_type=1)
            .move_to(0.5 * LEFT + 0 * UP)
            .scale(0.6)
            .set_fill(opacity=0.5, color=GREY_A)
        )
        cloud2 = (
            Cloud(cloud_type=2)
            .move_to(2 * RIGHT + 0.45 * UP)
            .scale(0.5)
            .set_fill(opacity=0.5, color=GREY_A)
        )
        cloud3 = (
            Cloud(cloud_type=2)
            .move_to(2.8 * LEFT + 0.35 * UP)
            .scale(0.4)
            .set_fill(opacity=0.5, color=GREY_A)
        )

        # define positions along a path for the satellite
        num_steps = 6
        sat_positions = np.linspace(
            LEFT * 1.7 + SAT_HEIGHT_SCENE_1 * UP,
            RIGHT * 1.7 + SAT_HEIGHT_SCENE_1 * UP,
            num_steps,
        )
        self.add(sat.move_to(sat_positions[0]))

        # Center of a half-circle on the ground where the center of the wave should land.
        # Otherwise, the waves look offset slightly
        radius = 0.4
        center = np.array([0, line_pos + 0.25, 0])

        theta_vals = np.linspace(0, np.pi, num_steps)
        target_points = list(
            reversed(
                [
                    center + radius * np.array([np.cos(t), np.sin(t), 0])
                    for t in theta_vals
                ]
            )
        )
        target_points[0] = target_points[0] + np.array(
            [
                0,
                0.2,
                0,
            ]
        )
        target_points[-1] = target_points[-1] + +np.array(
            [
                0,
                0.2,
                0,
            ]
        )

        for i, (sat_pos, target_point) in enumerate(zip(sat_positions, target_points)):
            if i >= 1:  # move satellite to a new position
                self.play(
                    sat.animate.move_to(sat_pos),
                    run_time=SAT_MOVE_RUNTIME,
                    rate_func=linear,
                )

            # forward wave
            start_wave = sat.get_center() + 0.5 * DOWN
            wave, anims = sat.send_beam(
                sat_initial_pos=start_wave,
                axes=axes,
                target_point=target_point,
                travel_direction="forward",
                amplitude=AMPLITUDE_DFLT,
                frequency=FREQUENCY_DFLT,
            )
            self.add(wave)
            self.play(*anims, run_time=BEAM_MOVE_RUNTIME, rate_func=linear)
            self.wait(TIME_BETWEEN_BEAMS)
            self.remove(wave)

            # backward wave: start at forward wave’s end, go back to satellite
            wave_back, anims_back = sat.send_beam(
                sat_initial_pos=target_point,
                axes=axes,
                target_point=tuple(start_wave[:2]),
                travel_direction="backward",
                amplitude=AMPLITUDE_DFLT,
                frequency=FREQUENCY_DFLT,
                x_range=[0, 1],
            )
            self.add(wave_back)
            if i in [0, 1, 5]:
                self.play(*anims_back, run_time=BEAM_MOVE_RUNTIME,
                          rate_func=linear)
            if i == 2:
                self.play(
                    *anims_back,
                    FadeIn(sun),
                    run_time=BEAM_MOVE_RUNTIME,
                    rate_func=linear,
                )
            if i == 3:
                self.play(
                    *anims_back,
                    FadeOut(sun),
                    FadeIn(moon),
                    run_time=BEAM_MOVE_RUNTIME,
                    rate_func=linear,
                )
            if i == 4:
                self.play(
                    *anims_back,
                    FadeIn(cloud1, cloud2, cloud3),
                    run_time=BEAM_MOVE_RUNTIME,
                    rate_func=linear,
                )
            self.remove(wave_back)

        sat.clear_updaters()
        self.play(
            FadeOut(
                *[sat, tree1, tree2, line, moon, cloud1,
                    cloud2, cloud3], run_time=0.4
            )
        )

        # text_sar = Paragraph("Synthetic Aperture", "Radar", alignment = 'center', color = BLACK).set_stroke(BLACK).set_fill(BLACK)
        # text_sar_short = Text("SAR", color = BLACK).set_stroke(BLACK).set_fill(BLACK)

        # self.play(Write(text_sar))
        # self.play(Transform(text_sar, text_sar_short))
        # self.play(FadeOut(text_sar))

        img = ImageMobject("./01_sar_bands/pngs/placeholder.png")

        img.stretch_to_fit_width(IMG_WIDTH)
        img.stretch_to_fit_height(IMG_HEIGHT)
        source = Text(
            "Oops, the image you're looking for might be excluded because of license restrictions.",
            font_size=13,
        ).set_color(SOURCE_COLOR)
        source.next_to(img, DOWN).align_to(img, LEFT)
        self.play(FadeIn(img, source), run_time=1.5)
        self.wait(4)
        self.play(FadeOut(img, source))

    def play_scene_2(self):

        # ValueTracker for time
        t_tracker = ValueTracker(0)

        # Sine wave function
        wavelengths = [30, 15, 7.5, 3.7, 2.4, 1.7, 1.1, 0.8]
        wavelengths_char = [
            "100-30cm",
            "30-15cm",
            "15-7.5cm",
            "7.5-3.8cm",
            "3.8-2.4cm",
            "2.4-1.7cm",
            "1.7-1.1cm",
            "1.1-0.8cm",
        ]
        names = ["P", "L", "S", "C", "X", "Ku", "K", "Ka"]

        ruler, tick_group = self.add_scale()

        waves = []
        for i, wavelength in enumerate(wavelengths):
            wave = self.add_wave(
                t_tracker=t_tracker,
                position=4.5 * LEFT + RIGHT * i * 1.3,
                frequency=(1 / wavelength) * 4,
                direction="backward",
            )
            self.add(wave)
            waves.append(wave)

        # Animate the tracker to make wave move
        self.play(t_tracker.animate.set_value(2), run_time=5, rate_func=linear)
        self.wait(1)

        # remove value tracker (otherwise waves bounce back)
        for wave in waves:
            wave.clear_updaters()

        self.play(*[wave.animate.shift(1.2 * DOWN) for wave in waves])

        title = (
            Text("SAR bands", weight=MEDIUM)
            .shift(TITLE_POS * UP)
            .set_stroke(BLACK)
            .set_fill(BLACK)
        )
        self.play(Write(title))

        labels = []
        for name, wave in zip(names, waves):

            # Place band name next to wave
            label = Text(name, font_size=40).set_color(BLACK)
            label.next_to(wave, UP, buff=0.5)

            # fade in level and make wave black again.
            self.play(
                AnimationGroup(
                    FadeIn(label).set_rate_func(linear),
                    wave.animate.scale(1.2).set_rate_func(there_and_back),
                ),
                run_time=0.5,
            )
            labels.append(label)

        for w in waves:
            w.clear_updaters()

        self.play(
            *[FadeOut(w) for w in waves],
            *[
                label.animate.shift(
                    RIGHT * (label.get_center() - ORIGIN)[0] * 0.18)
                for label in labels
            ],
        )

        # Transform the waves into the wavelength intervals
        char_objs = []
        for label, char in zip(labels, wavelengths_char):
            char_obj = (
                Text(char, font_size=25)
                .move_to(label.get_center() + 1 * DOWN)
                .set_color(BLACK)
            )
            char_objs.append(char_obj)

        line = Line(start=5 * LEFT, end=5 * RIGHT, color=BLACK).shift(DOWN)

        # Define multiple segments with (x1, x2) pairs
        segments = [
            (0 - 5, 7 - 5),  # P 100-30 cm
            (7 - 5, 7 + 1.5 - 5),  # L 30-15 cm
            (7 + 1.5 - 5, 8.5 + 0.75 - 5),  # S 15-7.5 cm
            (8.5 + 0.75 - 5, 8.5 + 0.75 + 0.15 - 5),  # C 7.5-3.8 cm
            (8.5 + 0.75 + 0.15 - 5, 8.5 + 0.75 + 0.15 + 0.14 - 5),  # X 3.8-2.4 cm
            (
                8.5 + 0.75 + 0.15 + 0.14 - 5,
                8.5 + 0.75 + 0.15 + 0.14 + 0.07 - 5,
            ),  # Ku 2.4-1.7 cm
            (
                8.5 + 0.75 + 0.15 + 0.14 + 0.07 - 5,
                8.5 + 0.75 + 0.15 + 0.14 + 0.07 + 0.03 + 0.15 - 5,
            ),  # K 1.7-1.1 cm
            (
                8.5 + 0.75 + 0.15 + 0.14 + 0.07 + 0.03 + 0.15 - 5,
                8.5 + 0.75 + 0.15 + 0.14 + 0.07 + 2 * 0.03 + 0.15 - 5,
            ),  # Ka 1.1-0.8 cm
        ]

        # Define the positions (in cm) along the line
        label_values = [100, 30, 15, 7, 0]  # cm

        x_positions = [
            segments[0][0],  # 100 cm (left)
            segments[0][1],  # 30 cm
            segments[1][1],  # 15 cm
            segments[2][1] + 0.05,  # 7 cm (with 10, 5cm too little space)
            line.get_end()[0],  # 0 cm (fixed right edge)
        ]

        # Place the labels
        line_label = []
        ticks = []
        tick_height = 0.2
        for x, val in zip(x_positions, label_values):
            tick = Line(
                start=np.array([x, line.get_y() - tick_height / 2, 0]),
                end=np.array([x, line.get_y() + tick_height / 2, 0]),
                color=BLACK,
            )
            label = (
                Text(f"{val}cm", font_size=24).set_color(
                    BLACK).set_stroke(color=BLACK)
            )
            label.next_to(np.array([x, line.get_y(), 0]), DOWN)
            line_label.append(label)
            ticks.append(tick)

        left_edges = []
        right_edges = []
        fill_areas = []
        for (x1, x2), label in zip(segments, labels):
            # Base line y-position
            y = line.get_center()[1]

            # Endpoints on the base line
            p1 = np.array([x1, y, 0])
            p2 = np.array([x2, y, 0])

            origin = label.get_bottom() + 0.15 * DOWN
            # Lines connecting origin to the segment ends
            left_line = Line(origin, p1, color=GRAY_D, stroke_width=4)
            right_line = Line(origin, p2, color=GRAY_D, stroke_width=4)

            # Fill the triangular region between them
            fill_area = Polygon(
                origin, p1, p2, color=GRAY, fill_opacity=0.3, stroke_opacity=0
            )

            # Add to the scene
            left_edges.append(left_line)
            right_edges.append(right_line)
            fill_areas.append(fill_area)

        self.play(FadeIn(line, *line_label, *ticks))
        self.play(FadeIn(*left_edges, *right_edges, *fill_areas))

        def rescale_segments(segments, scale_factors=None):
            """
            Rescale one or more segments by custom factors while keeping total length constant.

            Args:
                segments: list of (x1, x2) tuples
                scale_factors: dict {index: scale_factor}, e.g. {0: 0.6, 2: 1.2}
            """
            if scale_factors is None:
                scale_factors = {}

            lengths = np.array([x2 - x1 for x1, x2 in segments])
            total = lengths.sum()

            # Apply scale factors to specified segments
            scaled_lengths = lengths.copy()
            for i, s in scale_factors.items():
                scaled_lengths[i] *= s

            # Compute how much total length has changed
            scaled_total = scaled_lengths.sum()
            diff = total - scaled_total

            # Find indices not manually scaled
            unscaled_indices = [
                i for i in range(len(lengths)) if i not in scale_factors
            ]

            if unscaled_indices:
                # Distribute the leftover proportionally across unscaled segments
                unscaled_total = lengths[unscaled_indices].sum()
                correction_factor = 1 + diff / unscaled_total
                scaled_lengths[unscaled_indices] *= correction_factor
            else:
                # If all are scaled manually, just normalize to preserve total
                scaled_lengths *= total / scaled_lengths.sum()

            # Recompute new x-positions
            x_start = segments[0][0]
            x_positions = [x_start]
            for L in scaled_lengths:
                x_positions.append(x_positions[-1] + L)

            return list(zip(x_positions[:-1], x_positions[1:]))

        new_segments = rescale_segments(segments, {0: 0.4})

        # Compute all unique ticks from rescaled segments
        x_positions_new = (
            [new_segments[0][0]]
            + [x2 for (_, x2) in new_segments[:-1]]
            + [line.get_end()[0]]
        )

        new_ticks = []
        new_line_labels = []

        tick_height = 0.2
        y = line.get_y()

        for x, val in zip(x_positions_new, label_values):

            if val == 0:
                tick = Line(
                    start=np.array(
                        [line.get_end()[0], y - tick_height / 2, 0]),
                    end=np.array([line.get_end()[0], y + tick_height / 2, 0]),
                    color=BLACK,
                )
            else:
                tick = Line(
                    start=np.array([x, y - tick_height / 2, 0]),
                    end=np.array([x, y + tick_height / 2, 0]),
                    color=BLACK,
                )
            label = (
                Text(f"{val}cm", font_size=24).set_color(
                    BLACK).set_stroke(color=BLACK)
            )
            if val == 0:
                # anchor to line.get_end()
                label.next_to(np.array([line.get_end()[0], y, 0]), DOWN)
            else:
                label.next_to(np.array([x, y, 0]), DOWN)
            new_ticks.append(tick)
            new_line_labels.append(label)

        new_left_edges = []
        new_right_edges = []
        new_fill_areas = []

        for (x1, x2), label in zip(new_segments, labels):
            y = line.get_center()[1]
            p1 = np.array([x1, y, 0])
            p2 = np.array([x2, y, 0])
            origin = label.get_bottom() + 0.15 * DOWN

            new_left_edges.append(
                Line(origin, p1, color=GRAY_D, stroke_width=4))
            new_right_edges.append(
                Line(origin, p2, color=GRAY_D, stroke_width=4))
            new_fill_areas.append(
                Polygon(origin, p1, p2, color=GRAY,
                        fill_opacity=0.3, stroke_opacity=0)
            )

        x_start, x_end = new_segments[0]  # P segment
        seg_length = (x_end - x_start) / 3

        y = line.get_y()

        left_line = Line(
            start=np.array([x_start, y, 0]),
            end=np.array([x_start + seg_length, y, 0]),
            color=line.color,
        )

        right_line = Line(
            start=np.array([x_start + 2 * seg_length, y, 0]),
            end=line.get_end(),
            color=line.color,
        )

        gap_start = x_start + seg_length
        gap_end = x_start + 2 * seg_length
        fractions = np.linspace(0.32, 0.68, 3)
        dot_positions = [
            (gap_start + f * (gap_end - gap_start), y, 0) for f in fractions
        ]
        dots = [Dot(point=pos, radius=0.03, color=BLACK)
                for pos in dot_positions]

        self.play(
            FadeOut(line),
            FadeIn(left_line, right_line),
            *[FadeIn(dot) for dot in dots],
            *[Transform(old, new)
              for old, new in zip(left_edges, new_left_edges)],
            *[Transform(old, new)
              for old, new in zip(right_edges, new_right_edges)],
            *[Transform(old, new)
              for old, new in zip(fill_areas, new_fill_areas)],
            *[Transform(old, new) for old, new in zip(ticks, new_ticks)],
            *[Transform(old, new)
              for old, new in zip(line_label, new_line_labels)],
            run_time=2,
        )

        shift_amount = 3 * DOWN

        char_labels = []
        for label, char in zip(labels, wavelengths_char):
            char_label = Text(char, font_size=18, color=label.get_color())
            # Place it just below the main label
            char_label.next_to(label, DOWN, buff=0.05)  # small gap
            char_labels.append(char_label)
            self.add(char_label)

        shift_amount = 0.3 * UP
        animations = []
        for label, char_label in zip(labels, char_labels):
            animations.append(label.animate.shift(shift_amount).scale(0.75))
            animations.append(char_label.animate.shift(shift_amount))

        self.play(*[FadeIn(cl) for cl in char_labels], *animations, run_time=1)

        scale_factor = 1.3
        for label, char_label in zip(labels, char_labels):
            self.play(
                label.animate.scale(scale_factor).shift(0.1 * UP),
                char_label.animate.scale(scale_factor),
                run_time=0.2,
            )

            self.play(
                label.animate.scale(1 / scale_factor).shift(0.1 * DOWN),
                char_label.animate.scale(1 / scale_factor),
                run_time=0.2,
            )

        self.wait(3)
        self.play(
            FadeOut(
                title,
                *left_edges,
                *right_edges,
                *fill_areas,
                *ticks,
                *line_label,
                *labels,
                *char_labels,
                left_line,
                right_line,
                *dots,
            )
        )

    def play_scene_3(self):

        amplitude = 0.5
        frequency_tracker = ValueTracker(1 / 5)
        t_tracker = ValueTracker(0)
        wave = always_redraw(
            lambda: FunctionGraph(
                lambda x: amplitude
                * np.sin(
                    2
                    * np.pi
                    * frequency_tracker.get_value()
                    * (x + 0.5 * t_tracker.get_value())
                ),
                x_range=[-3, 3],
                color=BLACK,
                stroke_width=5,
            ).shift(DOWN)
        )

        title_lambda = (
            MathTex(r"\lambda = 100-30 \text{cm}", font_size=55)
            .set_color(BLACK)
            .move_to(TITLE_POS_WAVELENGTH * UP)
        )
        title_wavelength = (
            Text("P-band", font_size=60)
            .set_color(BLACK)
            .move_to(TITLE_POS_BAND_NAME * UP)
        )

        self.add(wave, title_lambda, title_wavelength)
        self.play(t_tracker.animate.set_value(8), run_time=4, rate_func=linear)

        new_title_lambda = (
            MathTex(r"\lambda = 30\text{–}15 \,\text{cm}", font_size=55)
            .set_color(BLACK)
            .move_to(TITLE_POS_WAVELENGTH * UP)
        )

        new_title_wavelength = (
            Text("L-band", font_size=60)
            .set_color(BLACK)
            .move_to(TITLE_POS_BAND_NAME * UP)
        )

        # Animate transformation
        self.play(
            Transform(title_lambda, new_title_lambda),
            Transform(title_wavelength, new_title_wavelength),
            frequency_tracker.animate.set_value(1 / 2),
            t_tracker.animate.set_value(8),
            run_time=1,
            rate_func=linear,
        )

        self.play(t_tracker.animate.set_value(
            16), run_time=4, rate_func=linear)

        self.play(
            FadeOut(
                new_title_lambda,
                new_title_wavelength,
                title_lambda,
                title_wavelength,
                wave,
            )
        )

    def play_scene_4(self, fw, fh):

        frequency = 1 / 0.8

        # Function to create ground with moving flats
        axes = Axes(
            x_range=[-fw / 2, fw / 2, 1],
            y_range=[-fh / 2, fh / 2, 1],
            x_length=fw,
            y_length=fh,
            tips=False,
        ).set_color(GREY)
        axes.add_coordinates(color=GREY)

        # Satellite
        sat_initial_pos = axes.c2p(-3, SAT_HEIGHT)
        sat = Sat().move_to(sat_initial_pos).scale(SCALE_SAT)

        # Dunes
        dune1 = Dune(
            amplitude=0.5,
            wavelength=1.5,
            shift=4,
            height_shift=GROUND_LINE_POS + 1,
            extent=[-4, 2],
            extent_dots=[-4, 2],
            y_bottom_dots=-1.7,
        )
        dune2 = Dune(
            amplitude=2,
            wavelength=3,
            shift=2,
            height_shift=GROUND_LINE_POS - 1,
            extent=[0.5, 4],
            extent_dots=[0.5, 4],
            y_bottom_dots=-1.7,
        )

        # Caves
        cave1 = (
            Cave(radius=0.6, color=LIGHT_GREY)
            .set_stroke(BLACK)
            .move_to(axes.c2p(-1, -3))
        )
        cave2 = (
            Cave(radius=0.5, color=LIGHT_GREY)
            .set_stroke(BLACK)
            .move_to(axes.c2p(1.7, -2.6))
        )

        self.add(sat, dune1, dune2)

        # animate stripmap mode
        num_steps = 4
        sat_positions = np.linspace(
            LEFT * 1.7 + SAT_HEIGHT * UP, RIGHT * 1.7 + SAT_HEIGHT * UP, num_steps
        )
        self.add(sat.move_to(sat_positions[0]))
        target_points = [sat_point + 4 * DOWN for sat_point in sat_positions]

        for i, (sat_pos, target_point) in enumerate(zip(sat_positions, target_points)):
            if i >= 1:  # move satellite to a new position
                self.play(
                    sat.animate.move_to(sat_pos),
                    run_time=SAT_MOVE_RUNTIME,
                    rate_func=linear,
                )

            # forward wave
            start_wave = sat.get_center() + 0.5 * DOWN
            wave, anims = sat.send_beam(
                sat_initial_pos=start_wave,
                axes=axes,
                target_point=target_point,
                travel_direction="forward",
                amplitude=0.3,
                frequency=frequency,
            )
            self.add(wave)
            self.play(*anims, run_time=BEAM_MOVE_RUNTIME_FAST,
                      rate_func=linear)
            self.wait(TIME_BETWEEN_BEAMS)
            self.remove(wave)

            # backward wave: start at forward wave’s end, go back to satellite
            wave_back, anims_back = sat.send_beam(
                sat_initial_pos=target_point,
                axes=axes,
                target_point=tuple(start_wave[:2]),
                travel_direction="backward",
                amplitude=0.3,
                frequency=frequency,
            )
            self.add(wave_back)
            if i in [0, 2]:
                self.play(
                    *anims_back, run_time=BEAM_MOVE_RUNTIME_FAST, rate_func=linear
                )
            elif i == 1:
                self.play(
                    *anims_back,
                    FadeIn(cave1),
                    run_time=BEAM_MOVE_RUNTIME_FAST,
                    rate_func=linear,
                )
            elif i == 3:
                self.play(
                    *anims_back,
                    FadeIn(cave2),
                    run_time=BEAM_MOVE_RUNTIME_FAST,
                    rate_func=linear,
                )
            self.remove(wave, wave_back)

        sat.clear_updaters()
        wave.clear_updaters()
        self.play(FadeOut(cave1, cave2, dune1, dune2, sat), run_time=0.5)
        self.remove(sat)

        # Second part: tree canopies

        # Base step points (axis coords)
        base_step_points = [
            (GROUND_LINE_INTERVAL[0], -2.4),
            (-2.6, -2),
            (-2, -2.5),
            (0, -2.5),
            (0, -2),
            (2.3, -2),
            (3, -1.5),
            (GROUND_LINE_INTERVAL[1], -1.5),
        ]
        flat_directions = [-1, 1, -1, 1]

        # Tracker to animate the movement of flat segments
        t = ValueTracker(0)
        # Ground VMobject that redraws each frame
        ground = always_redraw(
            lambda: self.create_moving_ground(
                base_step_points, axes, t.get_value(), flat_directions
            )
        )

        trees = [
            Tree(1).place_tree(axes=axes, ground=ground, x_axis=-3, scale=0.3),
            Tree(2).place_tree(axes=axes, ground=ground, x_axis=-0.95, scale=0.8),
            Tree(1, branch=True).place_tree(
                axes=axes, ground=ground, x_axis=0.6, scale=0.7
            ),
            Tree(2).place_tree(axes=axes, ground=ground, x_axis=1.5, scale=0.3),
            Tree(1).place_tree(axes=axes, ground=ground, x_axis=3.2, scale=0.5),
        ]

        num_steps = 4
        sat_positions = np.linspace(
            LEFT * 1.5 + SAT_HEIGHT * UP, RIGHT * 2 + SAT_HEIGHT * UP, num_steps
        )
        sat = Sat().move_to(sat_positions[0]).scale(SCALE_SAT)
        self.play(FadeIn(sat, ground, *trees))
        target_points = [sat_point + 3.8 * DOWN for sat_point in sat_positions]

        for i, (sat_pos, target_point) in enumerate(zip(sat_positions, target_points)):
            if i in [0, 1, 2]:  # move satellite to a new position
                self.play(
                    sat.animate.move_to(sat_pos),
                    run_time=SAT_MOVE_RUNTIME,
                    rate_func=linear,
                )

            if i in [0, 1]:
                # forward wave
                start_wave = sat.get_center() + 0.5 * DOWN
                wave, anims = sat.send_beam(
                    sat_initial_pos=start_wave,
                    axes=axes,
                    target_point=target_point,
                    travel_direction="forward",
                    amplitude=AMPLITUDE_DFLT,
                    frequency=frequency,
                )
                self.add(wave)
                self.play(*anims, run_time=BEAM_MOVE_RUNTIME_FAST,
                          rate_func=linear)
                self.wait(TIME_BETWEEN_BEAMS)
                self.remove(wave)

                # backward wave: start at forward wave’s end, go back to satellite
                wave_back, anims_back = sat.send_beam(
                    sat_initial_pos=target_point,
                    axes=axes,
                    target_point=tuple(start_wave[:2]),
                    travel_direction="backward",
                    amplitude=AMPLITUDE_DFLT,
                    frequency=frequency,
                )
                self.add(wave_back)
                self.play(
                    *anims_back, run_time=BEAM_MOVE_RUNTIME_FAST, rate_func=linear
                )

            elif i == 2:
                # forward wave only travels to branch which is approx. at (.75, -1)
                # override target point, a bit less down, s.t. only edge of
                # the wave touches the branch
                target_point = 0.75 * RIGHT + 0.8 * DOWN
                start_wave = sat.get_center() + 0.5 * DOWN
                wave, anims = sat.send_beam(
                    sat_initial_pos=start_wave,
                    axes=axes,
                    target_point=target_point,
                    travel_direction="forward",
                    amplitude=AMPLITUDE_DFLT,
                    frequency=frequency,
                )
                self.add(wave)
                self.play(*anims, run_time=0.8, rate_func=linear)
                self.wait(TIME_BETWEEN_BEAMS)
                self.remove(wave)
                # backward wave: 3 waves with different scattering 
                # and smaller amplitude
                # first wave
                wave_back, anims_back = sat.send_beam(
                    sat_initial_pos=target_point,
                    axes=axes,
                    target_point=tuple(start_wave[:2]),
                    travel_direction="backward",
                    amplitude=0.1,
                    frequency=frequency,
                )

                # second wave: bounce from branch -> trunk - trunk right tree
                # -> trunk tree -> satellite
                path_points = [
                    (1 * DOWN + 0.5 * RIGHT),
                    (1.75 * DOWN + 1.5 * RIGHT),
                    (1.5 * DOWN + 0.5 * RIGHT),
                    sat_positions[3] + 0.2 * DOWN,
                ]

                wave_back_bounce, _, t_tracker_bounce, pos_tracker_bounce = (
                    sat.send_beam_multipath(
                        path_points,
                        amplitude=0.1,
                        travel_direction="backward",
                        frequency=frequency,
                    )
                )

                # third wave: reflect away from the radar
                path_points = [target_point, (1 * LEFT + 1 * UP)]
                wave_away, anims_away, _, _ = sat.send_beam_multipath(
                    path_points,
                    amplitude=0.1,
                    travel_direction="backward",
                    frequency=frequency,
                )

                self.add(wave_back, wave_away, wave_back_bounce)
                self.play(
                    *anims_back,
                    *anims_away,
                    t_tracker_bounce.animate.set_value(3 * 0.65),
                    pos_tracker_bounce.animate.set_value(0.5),
                    run_time=0.8,
                    rate_func=linear,
                )
                wave_back.clear_updaters
                wave_away.clear_updaters
                self.remove(wave_back, wave_away)
                self.play(
                    t_tracker_bounce.animate.set_value(3),
                    pos_tracker_bounce.animate.set_value(1),
                    sat.animate.move_to(sat_positions[3]),
                    rate_func=linear,
                    run_time=1,
                )
                self.remove(wave_back_bounce)
                self.wait(0.5)

            self.remove(wave_back)

        sat.clear_updaters()
        self.remove(sat)

        # move trees and ground
        self.remove(trees[0], trees[2], trees[4])
        scale_tracker1 = ValueTracker(0.3)
        scale_tracker4 = ValueTracker(0.5)
        scale_tracker2 = ValueTracker(0.7)
        trees[0] = Tree(1).place_tree(
            axes=axes, ground=ground, x_axis=-3, scale=scale_tracker1
        )
        trees[2] = Tree(1, branch=True).place_tree(
            axes=axes, ground=ground, x_axis=0.6, scale=scale_tracker2
        )
        trees[4] = Tree(1).place_tree(
            axes=axes, ground=ground, x_axis=3.2, scale=scale_tracker4
        )
        self.add(trees[0], trees[2], trees[4])
        self.play(
            t.animate.set_value(1),  # move the flats
            scale_tracker1.animate.set_value(1.0),
            scale_tracker4.animate.set_value(0.4),
            scale_tracker2.animate.set_value(0.6),
            run_time=1.5,
        )

        self.play(
            t.animate.set_value(0.4),  # move the flats
            scale_tracker1.animate.set_value(0.4),
            scale_tracker4.animate.set_value(0.7),
            scale_tracker2.animate.set_value(0.8),
            run_time=1.4,
        )

        self.play(FadeOut(*trees, ground), run_time=0.5)

        img_esa_biomass = ImageMobject("./01_sar_bands/pngs/placeholder.png")
        img_esa_biomass.stretch_to_fit_height(IMG_HEIGHT)
        img_esa_biomass.stretch_to_fit_width(IMG_WIDTH)
        source = Text(
            "Oops, the image you're looking for might be excluded because of license restrictions.",
            font_size=15,
        ).set_color(SOURCE_COLOR)

        source.next_to(img_esa_biomass, DOWN).align_to(img_esa_biomass, LEFT)
        self.play(FadeIn(img_esa_biomass, source))
        self.wait(14)
        self.play(FadeOut(img_esa_biomass, shift=ORIGIN), FadeOut(source))

    def play_scene_5(self):

        amplitude = 0.5
        frequency_tracker = ValueTracker(1 / 3)
        t_tracker = ValueTracker(0)
        wave = always_redraw(
            lambda: FunctionGraph(
                lambda x: amplitude
                * np.sin(
                    2
                    * np.pi
                    * frequency_tracker.get_value()
                    * (x + 0.5 * t_tracker.get_value())
                ),
                x_range=[-3, 3],
                color=BLACK,
                stroke_width=5,
            ).shift(DOWN)
        )

        title_lambda = (
            MathTex(r"\lambda = 15-7.5 \text{cm}", font_size=55)
            .set_color(BLACK)
            .move_to(TITLE_POS_WAVELENGTH * UP)
        )
        title_wavelength = (
            Text("S-band", font_size=60)
            .set_color(BLACK)
            .move_to(TITLE_POS_BAND_NAME * UP)
        )

        self.add(wave, title_lambda, title_wavelength)
        self.play(t_tracker.animate.set_value(7), run_time=2, rate_func=linear)

        new_title_lambda = (
            MathTex(r"\lambda = 7.5\text{–}3.8 \,\text{cm}", font_size=55)
            .set_color(BLACK)
            .move_to(TITLE_POS_WAVELENGTH * UP)
        )

        new_title_wavelength = (
            Text("C-band", font_size=60)
            .set_color(BLACK)
            .move_to(TITLE_POS_BAND_NAME * UP)
        )

        # Animate transformation
        self.play(
            Transform(title_lambda, new_title_lambda),
            Transform(title_wavelength, new_title_wavelength),
            frequency_tracker.animate.set_value(1 / 1),
            t_tracker.animate.set_value(2),
            run_time=1,
            rate_func=linear,
        )
        self.play(t_tracker.animate.set_value(7), run_time=2, rate_func=linear)

        new_title_lambda2 = (
            MathTex(r"\lambda = 3.8\text{–}2.4 \,\text{cm}", font_size=55)
            .set_color(BLACK)
            .move_to(TITLE_POS_WAVELENGTH * UP)
        )

        new_title_wavelength2 = (
            Text("X-band", font_size=60)
            .set_color(BLACK)
            .move_to(TITLE_POS_BAND_NAME * UP)
        )

        # Animate transformation
        self.play(
            Transform(title_lambda, new_title_lambda2),
            Transform(title_wavelength, new_title_wavelength2),
            frequency_tracker.animate.set_value(2),
            t_tracker.animate.set_value(4),
            run_time=1,
            rate_func=linear,
        )
        self.play(t_tracker.animate.set_value(7), run_time=2, rate_func=linear)

        self.play(FadeOut(title_wavelength, title_lambda, wave), run_time=0.5)

    def play_scene_6(self, fw, fh):

        axes = Axes(
            x_range=[-fw / 2, fw / 2, 1],
            y_range=[-fh / 2, fh / 2, 1],
            x_length=fw,
            y_length=fh,
            tips=False,
        ).set_color(GREY)
        axes.add_coordinates(color=GREY)

        line = Line(
            axes.c2p(
                GROUND_LINE_INTERVAL[0], GROUND_LINE_POS
            ),  # convert axis coords → scene coords
            axes.c2p(GROUND_LINE_INTERVAL[1], GROUND_LINE_POS),
            color=BLACK,
        )
        self.add(line)

        house = House().scale(0.6).move_to(axes.c2p(-2, GROUND_LINE_POS), DOWN)
        car = Car().scale(0.5).move_to(axes.c2p(1, GROUND_LINE_POS), DOWN)
        tree1 = Tree().scale(0.7).move_to(axes.c2p(3.1, GROUND_LINE_POS), DOWN)
        tree2 = Tree().scale(0.3).move_to(axes.c2p(2.5, GROUND_LINE_POS), DOWN)

        self.add(house, tree1, tree2, car)

        # Satellite
        sat_initial_pos = axes.c2p(-2.8, SAT_HEIGHT)
        sat = Sat().move_to(sat_initial_pos).scale(SCALE_SAT)
        self.add(sat)

        # define positions along a path for the satellite
        num_steps = 4
        sat_positions = np.linspace(
            LEFT * 3.5 + SAT_HEIGHT * UP, 0.5 * LEFT + SAT_HEIGHT * UP, num_steps
        )
        self.add(sat.move_to(sat_positions[0]))

        # Center of a half-circle on the ground where the center of the wave should land.
        # Otherwise, the waves look offset slightly
        # points for the house
        radius = 0.5
        center = np.array([-2, -0.6, 0])

        theta_vals = np.linspace(0, np.pi, num_steps)
        target_points = list(
            reversed(
                [
                    center + radius * np.array([np.cos(t), np.sin(t), 0])
                    for t in theta_vals
                ]
            )
        )
        target_points[0] = target_points[0] + 0.3 * LEFT + 0.27 * UP
        target_points[3] = target_points[3] + 0.3 * RIGHT + 0.12 * UP

        # points for the car
        radius = 0.3
        center = np.array([1, -1.5, 0])

        theta_vals = np.linspace(0, np.pi, num_steps)
        target_points2 = list(
            reversed(
                [
                    center + radius * np.array([np.cos(t), np.sin(t), 0])
                    for t in theta_vals
                ]
            )
        )
        target_points2[0] = target_points2[0] + 0.3 * LEFT + 0.12 * UP
        target_points2[3] = target_points2[3] + 0.3 * RIGHT + 0.1 * UP

        sat_positions2 = np.linspace(
            0.2 * LEFT + SAT_HEIGHT * UP, 2 * RIGHT + SAT_HEIGHT * UP, num_steps
        )

        for _, (sat_pos, target_point) in enumerate(
            zip(
                np.append(sat_positions, sat_positions2, axis=0),
                [*target_points, *target_points2],
            )
        ):
            self.play(
                sat.animate.move_to(sat_pos),
                run_time=SAT_MOVE_RUNTIME,
                rate_func=linear,
            )

            # forward wave
            start_wave = sat.get_center() + 0.6 * DOWN
            wave, anims = sat.send_beam(
                sat_initial_pos=start_wave,
                axes=axes,
                target_point=target_point,
                travel_direction="forward",
                amplitude=AMPLITUDE_DFLT,
                frequency=FREQUENCY_X_BAND,
            )
            self.add(wave)
            self.play(*anims, run_time=BEAM_MOVE_RUNTIME_FAST,
                      rate_func=linear)
            self.wait(TIME_BETWEEN_BEAMS)
            self.remove(wave)

            # backward wave: start at forward wave’s end, go back to satellite
            wave_back, anims_back = sat.send_beam(
                sat_initial_pos=target_point,
                axes=axes,
                target_point=tuple(start_wave[:2]),
                travel_direction="backward",
                amplitude=AMPLITUDE_DFLT,
                frequency=FREQUENCY_X_BAND,
            )
            self.add(wave_back)
            self.play(*anims_back, run_time=BEAM_MOVE_RUNTIME_FAST,
                      rate_func=linear)
            self.remove(wave_back)

        sat.clear_updaters()
        self.play(
            FadeOut(*[sat, tree1, tree2, line, house, car], run_time=0.1))

        range_resolution = (
            MathTex(
                r"{\delta_{\text{az}} = }",  # left side
                r"{\lambda ",  # numerator
                r"\over" r"2 \, \Delta \,",
                r"\theta",  # denominator
                r"}",  # just to hold the fraction
                font_size=90,
            )
            .set_stroke(BLACK)
            .set_fill(BLACK)
        )
        # Grab numerator and denominator by LaTeX text
        self.play(Write(range_resolution))
        numerator = range_resolution.get_part_by_tex(r"\lambda")
        denominator = range_resolution.get_part_by_tex(r"\theta")

        # Highlight numerator
        title = (
            Text("Azimuth Resolution", font_size=50)
            .shift(TITLE_POS * UP)
            .set_stroke(BLACK)
            .set_fill(BLACK)
        )
        title_old = (
            Text("Azimuth Resolution", font_size=50)
            .shift(TITLE_POS * UP)
            .set_stroke(BLACK)
            .set_fill(BLACK)
        )
        title_flight = (
            Text("Flight Direction", font_size=50)
            .shift(TITLE_POS * UP)
            .set_stroke(BLACK)
            .set_fill(BLACK)
        )
        self.play(Write(title))
        self.wait(0.5)
        self.play(Transform(title, title_flight))
        self.wait(0.5)
        self.play(Transform(title, title_old))
        # self.play(FadeOut(title))

        wavelength_title = (
            Text("wavelength", font_size=30)
            .next_to(numerator, UP, buff=0.3)
            .set_stroke(TEAL_E)
            .set_fill(TEAL_E)
        )
        integration_title = (
            Text("integration angle", font_size=30)
            .next_to(denominator, DOWN, buff=0.3)
            .set_stroke(TEAL_E)
            .set_fill(TEAL_E)
        )
        self.play(Write(wavelength_title), run_time=0.5)
        self.play(numerator.animate.set_color(TEAL_E).scale(1.5))
        self.play(numerator.animate.scale(2 / 3))
        self.wait(5)
        self.play(FadeOut(wavelength_title),
                  numerator.animate.set_color(BLACK))

        self.play(Write(integration_title), run_time=0.5)
        self.play(denominator.animate.set_color(TEAL_E).scale(1.5))
        self.play(denominator.animate.scale(2 / 3))
        self.wait(1)
        self.play(FadeOut(integration_title),
                  denominator.animate.set_color(BLACK))
        self.wait(0.5)
        self.wait(6)

        self.play(FadeOut(range_resolution, title), run_time=0.5)

        img = ImageMobject("./01_sar_bands/pngs/placeholder.png")
        img.move_to(ORIGIN).scale(0.6)
        source = Text(
            "Oops, the image you're looking for might be excluded because of license restrictions.",
            font_size=15,
        ).set_color(SOURCE_COLOR)

        img.stretch_to_fit_width(IMG_WIDTH)
        img.stretch_to_fit_height(IMG_HEIGHT)

        source.next_to(img, DOWN).align_to(img, LEFT)
        self.play(FadeIn(img, source))
        self.wait(2)

        box = (
            Rectangle(width=3.2, height=1.1)
            .move_to(0.18 * DOWN + 0.32 * LEFT)
            .set_stroke(width=5, color=GOLD_D)
        )
        glow = create_glow(
            box,
            layers=6,
            start_stroke=5,
            scale_factor=1.2,
            color=GOLD_D,
            max_opacity=0.25,
        )
        box.set_fill(opacity=0)
        self.play(Create(VGroup(box, glow), lag_ratio=0))
        self.wait(2)
        self.play(FadeOut(img, box, glow, source))

    def play_scene_7(self):

        amplitude = 0.5
        frequency_tracker = ValueTracker(5)
        t_tracker = ValueTracker(0)
        shift_tracker_k = ValueTracker(0)
        wave = always_redraw(
            lambda: FunctionGraph(
                lambda x: amplitude
                * np.sin(
                    2
                    * np.pi
                    * frequency_tracker.get_value()
                    * (x + 0.5 * t_tracker.get_value())
                ),
                x_range=[-2.5, 2.5],
                color=BLACK,
                stroke_width=5,
            ).shift(shift_tracker_k.get_value() * LEFT + DOWN)
        )

        title_lambda = (
            MathTex(r"\lambda = 1.7-1.1 \text{cm}", font_size=60)
            .set_color(BLACK)
            .move_to(TITLE_POS_WAVELENGTH * UP)
        )
        title_k = (
            Text("K-band", font_size=60)
            .set_color(BLACK)
            .move_to(TITLE_POS_BAND_NAME * UP)
        )
        k_group = VGroup(title_k, title_lambda, wave)

        self.add(k_group)
        self.play(t_tracker.animate.set_value(4), run_time=4, rate_func=linear)
        self.play(t_tracker.animate.set_value(6), run_time=1, rate_func=linear)

        # Ka
        frequency_ku = 3
        wave_ku = always_redraw(
            lambda: FunctionGraph(
                lambda x: amplitude
                * np.sin(2 * np.pi * frequency_ku *
                         (x + 0.5 * t_tracker.get_value())),
                x_range=[-2.5, 2.5],
                color=BLACK,
                stroke_width=5,
            ).shift(DOWN + 3 * LEFT)
        )
        title_lambda_ku = (
            MathTex(r"\lambda = 2.4-1.7 \text{cm}", font_size=60)
            .set_color(BLACK)
            .move_to(3 * LEFT + TITLE_POS_WAVELENGTH * UP)
        )
        title_ku = (
            Text("Ku", font_size=60)
            .set_color(BLACK)
            .move_to(4 * LEFT + TITLE_POS_BAND_NAME * UP)
        )
        title_band_ku = (
            Text("-band", font_size=60)
            .set_color(BLACK)
            .next_to(title_ku, RIGHT, buff=0.1)
        )
        ku_group = VGroup(wave_ku, title_lambda_ku, title_ku, title_band_ku)

        # Ku
        frequency_ka = 7
        wave_ka = always_redraw(
            lambda: FunctionGraph(
                lambda x: amplitude
                * np.sin(2 * np.pi * frequency_ka *
                         (x + 0.5 * t_tracker.get_value())),
                x_range=[-2.5, 2.5],
                color=BLACK,
                stroke_width=5,
            ).shift(DOWN + 3.5 * RIGHT)
        )
        title_ka = (
            Text("Ka-band", font_size=60)
            .set_color(BLACK)
            .move_to(3.5 * RIGHT + TITLE_POS_BAND_NAME * UP)
        )
        title_lambda_ka = (
            MathTex(r"\lambda = 1.1-0.8 \text{cm}", font_size=60)
            .set_color(BLACK)
            .move_to(3.5 * RIGHT + TITLE_POS_WAVELENGTH * UP)
        )
        ka_group = VGroup(wave_ka, title_lambda_ka, title_ka)

        self.play(k_group.animate.shift(3.5 * RIGHT),
                  run_time=0.5, rate_func=linear)
        shift_tracker_k.set_value(-3.5)
        self.play(FadeIn(ku_group), run_time=0.5, rate_func=linear)
        self.play(t_tracker.animate.set_value(4), run_time=3, rate_func=linear)
        self.play(FadeOut(ku_group), run_time=0.1, rate_func=linear)
        self.remove(ku_group)

        self.play(k_group.animate.shift(7 * LEFT),
                  run_time=0.5, rate_func=linear)
        shift_tracker_k.set_value(3.5)
        self.play(FadeIn(ka_group), run_time=0.5, rate_func=linear)
        self.play(t_tracker.animate.set_value(8), run_time=3, rate_func=linear)

        self.play(FadeOut(ka_group, k_group))

    def play_scene_8(self, fw, fh):

        cloud2 = (
            Cloud(cloud_type=2, stroke_w=5)
            .scale(1.5)
            .shift(0.5 * UP)
            .set_fill(opacity=0.5, color=GREY_A)
        )

        raindrops = [
            Raindrop(stroke_w=6)
            .scale(0.05)
            .shift(0.75 * DOWN + 0.5 * RIGHT)
            .set_fill(opacity=0.5, color=GREY_A),
            Raindrop(stroke_w=6)
            .scale(0.05)
            .shift(0.75 * DOWN + 0.1 * LEFT)
            .set_fill(opacity=0.5, color=GREY_A),
            Raindrop(stroke_w=6)
            .scale(0.05)
            .shift(0.75 * DOWN + 0.9 * LEFT)
            .set_fill(opacity=0.5, color=GREY_A),
            Raindrop(stroke_w=6)
            .scale(0.05)
            .shift(0.75 * DOWN + 0.4 * LEFT)
            .set_fill(opacity=0.5, color=GREY_A),
            Raindrop(stroke_w=6)
            .scale(0.05)
            .shift(0.75 * DOWN + 0.9 * RIGHT)
            .set_fill(opacity=0.5, color=GREY_A),
        ]

        self.add(cloud2, *raindrops)

        # Define vertical range
        fall_distance = 1

        # Each drop has a phase offset (different starting point in sawtooth)
        offsets = [0, 0.8, 0.2, 0.5, 0.7]  # fractions of the cycle
        trackers = [ValueTracker(0) for _ in raindrops]

        # track phases and start positions
        start_positions = [np.array(drop.get_center()) for drop in raindrops]
        last_phases = [offset for offset in offsets]

        for i, (drop, tracker, offset, start_pos) in enumerate(
            zip(raindrops, trackers, offsets, [
                drop.get_center() for drop in raindrops])
        ):

            def updater(m, t=tracker, s=start_pos, o=offset, idx=i):
                phase = (t.get_value() + o) % 1

                # slightly move drops to the left and right in each cycle
                if last_phases[idx] > 0.9 and phase < 0.1:
                    dx = random.uniform(-0.1, 0.1)
                    start_positions[idx] = np.array(
                        start_positions[idx]) + RIGHT * dx

                last_phases[idx] = phase
                new_pos = start_positions[idx] + DOWN * phase * fall_distance
                m.move_to(new_pos)
                m.set_stroke(opacity=1 - phase)

            drop.add_updater(updater)

        # Play rain
        self.play(
            *[tracker.animate.set_value(5) for tracker in trackers],
            run_time=5,
            rate_func=linear,
        )

        for drop in raindrops:
            drop.clear_updaters()
        self.play(FadeOut(cloud2, *raindrops))

        cloud1 = (
            Cloud(cloud_type=1, stroke_w=5)
            .scale(0.7)
            .shift(0.8 * UP + 1.5 * RIGHT)
            .set_fill(opacity=0.5, color=GREY_A)
        )
        cloud2 = (
            Cloud(cloud_type=2, stroke_w=5)
            .scale(0.7)
            .shift(1 * UP + 1.3 * LEFT)
            .set_fill(opacity=0.5, color=GREY_A)
        )

        # Axes that exactly match the frame
        axes = Axes(
            x_range=[-fw / 2, fw / 2, 1],
            y_range=[-fh / 2, fh / 2, 1],
            x_length=fw,
            y_length=fh,
            tips=False,
        ).set_color(GREY)
        axes.add_coordinates(color=GREY)

        line = Line(
            axes.c2p(
                GROUND_LINE_INTERVAL[0], GROUND_LINE_POS
            ),  # convert axis coords → scene coords
            axes.c2p(GROUND_LINE_INTERVAL[1], GROUND_LINE_POS),
            color=BLACK,
        )

        tree1 = Tree(tree_type=1).scale(0.6).align_to(
            line, DOWN).shift(2.7 * LEFT)
        tree2 = Tree(tree_type=2).scale(
            0.7).align_to(line, DOWN).shift(3 * RIGHT)
        tree3 = Tree(tree_type=2).scale(
            0.3).align_to(line, DOWN).shift(2 * RIGHT)

        self.play(FadeIn(tree1, tree2, tree3, line))

        # Snow cover imaging (Cryosat inspired)
        def func(x):
            """Shape of a smooth snow cover"""
            return (
                -(np.sin(3 * x + 0.5) * 0.1 + 1) *
                (x + 3.5) * (x - 3.5) * 0.07 + 0.9
            ) * 0.5

        snow_line = axes.plot(func, x_range=GROUND_LINE_INTERVAL, color=WHITE)
        # Fill area under the curve from x=-2 to x=2
        snow_cover = axes.get_area(
            snow_line, x_range=GROUND_LINE_INTERVAL, color=WHITE, opacity=0.8
        )

        # Shift the area up by 1 unit
        snow_line.shift(GROUND_LINE_POS * UP)
        snow_cover.shift(GROUND_LINE_POS * UP)

        # Show axes and function
        self.play(Create(snow_line), FadeIn(snow_cover))

        # Satellite
        sat_initial_pos = axes.c2p(-2.8, SAT_HEIGHT)
        sat = Sat().move_to(sat_initial_pos)
        self.add(sat)

        # define positions along a path for the satellite
        num_steps = 4
        sat_positions = np.linspace(
            2 * LEFT + SAT_HEIGHT * UP, 2 * RIGHT + SAT_HEIGHT * UP, num_steps
        )
        self.add(sat.move_to(sat_positions[0]))

        # Center of a half-circle on the ground where the center of the wave should land.
        # Otherwise, the waves look offset slightly
        # points for the house
        radius = 0.3
        center = np.array([0.2, GROUND_LINE_POS + 1.1, 0])

        theta_vals = np.linspace(0, np.pi, num_steps)
        target_points = list(
            reversed(
                [
                    center + radius * np.array([np.cos(t), np.sin(t), 0])
                    for t in theta_vals
                ]
            )
        )
        target_points[0] = target_points[0] + 0.3 * LEFT + 0.2 * UP
        target_points[3] = target_points[3] + 0.3 * RIGHT + 0.2 * UP

        for _, (sat_pos, target_point) in enumerate(zip(sat_positions, target_points)):
            self.play(
                sat.animate.move_to(sat_pos),
                run_time=SAT_MOVE_RUNTIME,
                rate_func=linear,
            )

            # forward wave
            start_wave = sat.get_center() + 0.5 * DOWN
            wave, anims = sat.send_beam(
                sat_initial_pos=start_wave,
                axes=axes,
                target_point=target_point,
                travel_direction="forward",
                amplitude=0.25,
                frequency=FREQUENCY_K_BAND,
            )
            self.add(wave)
            self.play(*anims, run_time=BEAM_MOVE_RUNTIME_FAST,
                      rate_func=linear)
            self.wait(TIME_BETWEEN_BEAMS)
            self.remove(wave)

            # backward wave: start at forward wave’s end, go back to satellite
            wave_back, anims_back = sat.send_beam(
                sat_initial_pos=target_point,
                axes=axes,
                target_point=tuple(start_wave[:2]),
                travel_direction="backward",
                amplitude=0.25,
                frequency=1 / 0.2,
            )
            self.add(wave_back)
            self.play(*anims_back, run_time=BEAM_MOVE_RUNTIME_FAST,
                      rate_func=linear)
            self.wait(TIME_BETWEEN_BEAMS)
            self.remove(wave_back)

        sat.clear_updaters()
        self.play(FadeOut(sat, run_time=0.1))
        self.play(FadeOut(snow_line, snow_cover), run_time=0.5)

        shift_up1 = 0.4
        shift_up2 = 0.1
        raindrops = [
            Raindrop(stroke_w=6).scale(0.02).shift(
                shift_up2 * UP + 1.2 * RIGHT),
            Raindrop(stroke_w=6).scale(0.02).shift(
                shift_up2 * UP + 1.9 * RIGHT),
            Raindrop(stroke_w=6).scale(0.02).shift(
                shift_up2 * UP + 1.5 * RIGHT),
            Raindrop(stroke_w=6).scale(0.02).shift(
                shift_up1 * UP + 1.3 * LEFT),
            Raindrop(stroke_w=6).scale(0.02).shift(
                shift_up1 * UP + 1.7 * LEFT),
            Raindrop(stroke_w=6).scale(0.02).shift(
                shift_up1 * UP + 0.9 * LEFT),
        ]
        self.play(FadeIn(cloud1, cloud2, *raindrops), run_time=0.5)

        # Define vertical range
        fall_distance = 0.65

        # Each drop has a phase offset (different starting point in sawtooth)
        offsets = [0, 0.8, 0.2, 0.5, 0.7, 0.9]  # fractions of the cycle
        trackers = [ValueTracker(0) for _ in raindrops]

        # track phases and start positions
        start_positions = [np.array(drop.get_center()) for drop in raindrops]
        last_phases = [offset for offset in offsets]

        for i, (drop, tracker, offset, start_pos) in enumerate(
            zip(raindrops, trackers, offsets, [
                drop.get_center() for drop in raindrops])
        ):

            def updater(m, t=tracker, s=start_pos, o=offset, idx=i):
                phase = (t.get_value() + o) % 1

                # slightly move drops to the left and right in each cycle
                if last_phases[idx] > 0.9 and phase < 0.1:
                    dx = random.uniform(-0.1, 0.1)
                    start_positions[idx] = np.array(
                        start_positions[idx]) + RIGHT * dx

                last_phases[idx] = phase
                new_pos = start_positions[idx] + DOWN * phase * fall_distance
                m.move_to(new_pos)
                m.set_stroke(opacity=1 - phase)

            drop.add_updater(updater)

        self.play(
            *[tracker.animate.set_value(2) for tracker in trackers],
            run_time=2,
            rate_func=linear,
        )
        self.play(FadeOut(cloud1, cloud2, *raindrops), run_time=0.5)
        self.remove(cloud1, cloud2, *raindrops)

        # Running car
        sat_initial_pos = axes.c2p(-2.8, 3)
        sat = Sat().move_to(sat_initial_pos)

        car = Car().scale(0.5).align_to(line, DOWN).shift(0.5 * LEFT)
        self.play(FadeIn(car))
        self.wait(0.1)  # forces a frame render
        base_pos = car.get_center().copy()

        car.add_updater(
            lambda m, dt: m.move_to(
                base_pos + UP * 0.01 * (np.sin(60 * self.time) + 1))
        )

        smokes = []
        animations = []

        num_smoke = 20
        for i in range(num_smoke):
            smoke = Circle(radius=0.03, color=None,
                           fill_opacity=0.7, stroke_opacity=0)
            smoke.move_to(car.get_left() + DOWN * 0.1)
            # self.add(smoke)
            smokes.append(smoke)

            animations.append(
                smoke.animate.set_color(GREY)
                .shift(
                    UP * 0.05 * (num_smoke - i)
                    + LEFT
                    * random.uniform(0.05 * (num_smoke - i), 0.07 * (num_smoke - i))
                )
                .scale(1 * (num_smoke - i))
                .set_fill(opacity=0)
            )

        # play all animations at once with small lag between them
        self.play(
            AnimationGroup(*animations, lag_ratio=0.1,
                           run_time=3, rate_func=linear)
        )
        car.clear_updaters()

        # clean up
        for smoke in smokes:
            smoke.remove()

        self.play(FadeOut(tree1, tree2, tree3, line, car), run_time=0.5)
        self.remove(sat)

        img_ka_band = ImageMobject(
            "./01_sar_bands/pngs/placeholder"
        )
        img_ka_band.move_to(ORIGIN).scale(0.5)
        img_ka_band.stretch_to_fit_width(IMG_WIDTH)
        img_ka_band.stretch_to_fit_height(IMG_HEIGHT)
        source = Text(
            "Oops, the image you're looking for might be excluded because of license restrictions.",
            font_size=15
        ).set_color(SOURCE_COLOR)

        source.next_to(img_ka_band, DOWN).align_to(img_ka_band, LEFT)
        self.play(FadeIn(img_ka_band, source))
        self.wait(2)

        box = Rectangle(width=3, height=2, stroke_width=5, color=GOLD_D).move_to(
            0.15 * LEFT + 0.4 * UP
        )
        glow = create_glow(
            box,
            layers=6,
            start_stroke=5,
            scale_factor=1.2,
            color=GOLD_D,
            max_opacity=0.25,
        )
        box.set_stroke(GOLD_D, 8)
        box.set_fill(opacity=0)
        self.play(FadeIn(VGroup(box, glow)))
        self.wait(2)

        img_ka_band2 = ImageMobject(
            "./01_sar_bands/pngs/placeholder.png"
        )
        img_ka_band2.move_to(ORIGIN).scale(0.5)
        img_ka_band2.stretch_to_fit_width(IMG_WIDTH)
        img_ka_band2.stretch_to_fit_height(IMG_HEIGHT)
        box2 = Rectangle(width=4.5, height=3.6, stroke_width=5).move_to(
            1.8 * RIGHT + 1.1 * DOWN
        )  # .rotate(-25*DEGREES)
        glow2 = create_glow(
            box2,
            layers=6,
            start_stroke=5,
            scale_factor=1.2,
            color=GOLD_D,
            max_opacity=0.25,
        )
        box2.set_stroke(GOLD_D, 8)
        box2.set_fill(opacity=0)

        self.play(FadeOut(img_ka_band, VGroup(box, glow)),
                  FadeIn(img_ka_band2))
        self.wait(2)
        self.play(FadeIn(VGroup(box2, glow2)))
        self.wait(2)
        self.play(FadeOut(img_ka_band2, source, VGroup(box2, glow2)))

    def play_scene_9(self):

        shift_down = 0.5
        images = [
            ImageMobject(
                "./01_sar_bands/pngs/placeholder.png"),
            ImageMobject(
                "./01_sar_bands/pngs/placeholder.png"),
            ImageMobject(
                "./01_sar_bands/pngs/placeholder.png")
        ]

        images_scaled = [img.copy() for img in images]
        for img in images_scaled:
            img.stretch_to_fit_width(3.8)
            img.stretch_to_fit_height(2.5)
        images_group = Group(*images_scaled).arrange(RIGHT, buff=0.8)

        titles = [
            Text("S", font_size=40),
            Text("L", font_size=40),
            Text("P", font_size=40),
        ]

        for title, img in zip(titles, images_scaled):
            title.set_color(BLACK).set_stroke(color=BLACK)
            title.next_to(img, UP)

        source = Text(
            "Oops, the image you're looking for might be excluded because of license restrictions.",
            font_size=15,
        ).set_color(SOURCE_COLOR)
        source.next_to(images_group, DOWN)  # .align_to(images_group, LEFT)

        self.play(FadeIn(images_group, source))
        self.play(Write(titles[0]))
        self.play(Write(titles[1]))
        self.play(Write(titles[2]))

        for img in images_group:
            self.play(
                # scale up and slightly move up
                img.animate.scale(1.1), run_time=0.5
            )

            self.play(img.animate.scale(1 / 1.1), run_time=0.5)

        self.play(*[title.animate.shift(1.2 * UP) for title in titles])
        self.play(
            FadeOut(images_scaled[0], images_scaled[1],
                    images_scaled[2], source),
            titles[1]
            .animate.set_color(FADE_OUT_COLOR)
            .set_fill(opacity=0.6)
            .set_stroke(color=FADE_OUT_COLOR, opacity=0.6),
            titles[2]
            .animate.set_color(FADE_OUT_COLOR)
            .set_fill(opacity=0.6)
            .set_stroke(color=FADE_OUT_COLOR, opacity=0.6),
            run_time=0.5,
        )

        for img in images:
            img.shift(shift_down * DOWN)

        # Stretch all images to the same width and height
        current_height = images[0].get_height()
        current_width = images[0].get_width()
        for img in images:
            img.stretch_to_fit_width(IMG_WIDTH)
            img.stretch_to_fit_height(5.5)

        new_height = images[0].get_height()
        new_width = images[0].get_width()


        img_S_band, img_L_band, img_P_band = images

        source = Text(
            "Oops, the image you're looking for might be excluded because of license restrictions.",
            font_size=15,
        ).set_color(SOURCE_COLOR)

        source.next_to(img_S_band, DOWN).align_to(img_S_band, LEFT)
        self.play(FadeIn(img_S_band, source))
        self.wait(2)

        airport_badge = (
            Text("AIRPORT", font_size=25)
            .set_color(BLACK)
            .move_to(0.5 * DOWN + 1.5 * RIGHT)
            .move_to(1.5 * UP + 3 * RIGHT)
        )
        box = SurroundingRectangle(
            airport_badge, color=BORDER_COLOR_BOX, buff=0.1, stroke_width=2
        ).set_fill(LIGHT_BEIGE, opacity=1)
        badge_group = VGroup(box, airport_badge)

        self.play(FadeIn(badge_group))
        self.wait(1)
        self.play(FadeOut(badge_group))

        snow_badge = (
            Text("SNOW COVER", font_size=25)
            .set_color(BLACK)
            .move_to(1 * DOWN + 2 * LEFT)
        )
        box2 = SurroundingRectangle(
            snow_badge, color=BORDER_COLOR_BOX, buff=0.1, stroke_width=2
        ).set_fill(LIGHT_BEIGE, opacity=1)
        snow_group = VGroup(box2, snow_badge)

        self.play(FadeIn(snow_group))
        self.wait(1)
        self.play(FadeOut(snow_group))

        self.play(
            FadeOut(img_S_band),
            FadeIn(img_L_band),
            titles[0]
            .animate.set_color(FADE_OUT_COLOR)
            .set_fill(opacity=FADE_OUT_OPACITY)
            .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY),
            titles[1]
            .animate.set_color(BLACK)
            .set_fill(opacity=1)
            .set_stroke(color=BLACK, opacity=1),
            titles[2]
            .animate.set_color(FADE_OUT_COLOR)
            .set_fill(opacity=FADE_OUT_OPACITY)
            .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY),
        )
        self.wait(2)

        self.play(
            FadeOut(img_L_band),
            FadeIn(img_P_band),
            titles[0]
            .animate.set_color(FADE_OUT_COLOR)
            .set_fill(opacity=FADE_OUT_OPACITY)
            .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY),
            titles[1]
            .animate.set_color(FADE_OUT_COLOR)
            .set_fill(opacity=FADE_OUT_OPACITY)
            .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY),
            titles[2]
            .animate.set_color(BLACK)
            .set_fill(opacity=1)
            .set_stroke(color=BLACK, opacity=1),
        )
        self.wait(2)

        self.play(FadeOut(img_P_band, source, *titles))

    def play_scene_10(self):

        # Table with band names and wavelengths on left hand side
        names = ["P", "L", "S", "C", "X", "Ku", "K", "Ka"]
        wavelengths_char = [
            "100-30cm",
            "30-15cm",
            "15-7.5cm",
            "7.5-3.8cm",
            "3.8-2.4cm",
            "2.4-1.7cm",
            "1.7-1.1cm",
            "1.1-0.8cm",
        ]

        table_line = Line(start=[-2.5, 2.5, 0], end=[2.5, 2.5, 0], color=BLACK)

        char_objs = []
        for char in reversed(wavelengths_char):
            char_obj = Text(char, font_size=40).set_color(BLACK)
            char_objs.append(char_obj)

        labels = []
        for name in reversed(names):
            label = Text(name, font_size=40).set_color(BLACK)
            labels.append(label)

        scale_factor = 0.8
        labels_x = -1.5
        target_ys = np.linspace(2, -3, len(labels))
        chars_x = 1
        for i in range(len(char_objs)):
            char_objs[i].move_to([chars_x, target_ys[i], 0]
                                 ).scale(scale_factor)
            labels[i].move_to([labels_x, target_ys[i], 0]).scale(scale_factor)

        # Column titles
        bands_title = (
            Text("Bands", font_size=40)
            .next_to(labels[0], UP, buff=0.5)
            .set_color(BLACK)
        )
        lambda_title = (
            MathTex(r"\lambda", font_size=45)
            .next_to(char_objs[0], UP, buff=0.5)
            .set_color(BLACK)
        )

        # Animate titles fading in
        self.play(
            FadeIn(bands_title, lambda_title, table_line, *labels, *char_objs),
        )
        self.wait(3)

        for label, char_ob in zip(labels, char_objs):
            self.play(
                label.animate.scale(1.2), char_ob.animate.scale(1.2), run_time=0.3
            )
            self.play(
                label.animate.scale(0.83), char_ob.animate.scale(0.83), run_time=0.3
            )

        # transform each label to A, B, C, ...
        labels_easy_char = ["A", "B", "C", "D", "E", "F", "G", "H"]
        labels_easy = []
        for i, name in enumerate(labels_easy_char):
            label = (
                Text(name, font_size=40)
                .set_color(RED_E)
                .move_to([labels_x, target_ys[i], 0])
                .scale(scale_factor)
            )
            labels_easy.append(label)

        original_labels = [label.copy() for label in labels]
        self.play(
            *[
                Transform(label, label_easy)
                for label, label_easy in zip(labels, labels_easy)
            ],
            run_time=2,
        )
        self.wait(2)
        self.play(
            *[
                Transform(label, original)
                for label, original in zip(labels, original_labels)
            ],
            run_time=2,
        )

        # scale closer to each other
        shift_left = 3.5
        self.play(
            item.animate.shift(shift_left * LEFT)
            for item in [*labels, *char_objs, bands_title, lambda_title, table_line]
        )
        new_target_ys = np.linspace(
            1.75, -2.25, len(labels))  # tighter spacing
        vertical_shift = new_target_ys[0] - target_ys[0]  # shift for titles

        # Animate movement
        self.play(
            *[
                labels[i].animate.move_to(
                    [labels_x - shift_left, new_target_ys[i], 0])
                for i in range(len(labels))
            ],
            *[
                char_objs[i].animate.move_to(
                    [chars_x - shift_left, new_target_ys[i], 0]
                )
                for i in range(len(char_objs))
            ],
            bands_title.animate.shift(UP * vertical_shift),
            lambda_title.animate.shift(UP * vertical_shift),
            table_line.animate.shift(UP * vertical_shift),
            run_time=1,
        )
        self.wait(2)

        doc = SecretDocument().move_to(3 * RIGHT).scale(0.8)
        self.play(FadeIn(doc.paper, doc.fold))
        self.play(LaggedStart(*[Create(line)
                  for line in doc.lines], lag_ratio=0.1))
        self.play(FadeIn(doc.secret_group))
        self.wait()

        germany_flag = (
            ImageMobject("./01_sar_bands/pngs/germany_flag.png")
            .scale(0.05)
            .next_to(doc, direction=UP + RIGHT, buff=0.2)
        )
        uk_flag = (
            ImageMobject("./01_sar_bands/pngs/uk_flag.png")
            .scale(0.05)
            .next_to(doc, direction=UP + LEFT, buff=0.2)
        )
        us_flag = (
            ImageMobject("./01_sar_bands/pngs/us_flag.png")
            .scale(0.05)
            .next_to(doc, direction=DOWN, buff=0.4)
        )

        self.play(FadeIn(germany_flag), run_time=0.3)
        self.play(FadeIn(uk_flag), run_time=0.3)
        self.play(FadeIn(us_flag), run_time=0.3)

        self.play(
            FadeOut(
                doc.paper,
                *doc.lines,
                doc.fold,
                doc.secret_group,
                germany_flag,
                uk_flag,
                us_flag,
            )
        )

        text_ieee1 = Text(
            "IEEE radar bands", font_size=50, color=BLACK, stroke_color=BLACK
        ).move_to(3 * RIGHT + 0.5 * UP)
        text_ieee2 = Text(
            "US Institute of Electrical\nand Electronics Engineers",
            font_size=30,
            color=BLACK,
            stroke_color=BLACK,
            slant=ITALIC,
        ).next_to(text_ieee1, direction=DOWN, buff=0.5)
        self.play(FadeIn(text_ieee1), run_time=1)
        self.wait(1)
        self.play(FadeIn(text_ieee2), run_time=1)
        self.wait(2)
        self.play(FadeOut(text_ieee1, text_ieee2))

        # FadeOut all but P
        FONT_SIZE = 60
        text_P = Text("P", font_size=FONT_SIZE).set_color(
            BLACK).move_to(3 * RIGHT)
        text_transform = Text("Experimental", font_size=FONT_SIZE, color=BLACK).move_to(
            3 * RIGHT
        )

        t_tracker = ValueTracker(0.0)
        amplitude = 0.2

        wavelengths = [30, 15, 7.5, 3.7, 2.4, 1.7, 1.1, 0.8]
        wavelengths_char = [
            "100-30cm",
            "30-15cm",
            "15-7.5cm",
            "7.5-3.8cm",
            "3.8-2.4cm",
            "2.4-1.7cm",
            "1.7-1.1cm",
            "1.1-0.8cm",
        ]
        names = ["P", "L", "S", "C", "X", "Ku", "K", "Ka"]

        def sine_wave(x):
            return amplitude * np.sin(
                2 * np.pi * (1 / wavelengths[0]) *
                4 * (x + 0.5 * t_tracker.get_value())
            )

        # Wave always redraws (behind the letter)
        wave1 = always_redraw(
            lambda: FunctionGraph(sine_wave, x_range=[
                                  1, 2.65, 0.01], color=BLACK)
        )
        wave2 = always_redraw(
            lambda: FunctionGraph(sine_wave, x_range=[
                                  3.35, 5, 0.01], color=BLACK)
        )

        # Add wave behind the letter
        self.add(wave1, wave2)

        # Animate wave movement
        # Position relative to `text_P`
        self.play(
            *[
                label.animate.set_color(FADE_OUT_COLOR)
                .set_fill(opacity=FADE_OUT_OPACITY)
                .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY)
                for label in labels[:-1]
            ],
            *[
                char_obj.animate.set_color(FADE_OUT_COLOR)
                .set_fill(opacity=FADE_OUT_OPACITY)
                .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY)
                for char_obj in char_objs[:-1]
            ],
            FadeIn(text_P, wave1, wave2),
            run_time=1,
        )
        self.play(
            t_tracker.animate.set_value(10),
            run_time=3,
            rate_func=linear,
        )
        self.play(FadeOut(wave1, wave2))
        self.play(Transform(text_P, text_transform))
        self.wait(2)
        self.play(FadeOut(text_P))

        # FadeOut all but L, S
        text_L = Text("L", font_size=FONT_SIZE).set_color(
            BLACK).move_to(3 * RIGHT + UP)
        text_S = (
            Text("S", font_size=FONT_SIZE).set_color(
                BLACK).move_to(3 * RIGHT + DOWN)
        )

        text_L_transform = (
            Text("Long", font_size=FONT_SIZE).set_color(
                BLACK).move_to(3 * RIGHT + UP)
        )
        text_S_transform = (
            Text("Short", font_size=FONT_SIZE)
            .set_color(BLACK)
            .move_to(3 * RIGHT + DOWN)
        )

        t_tracker.set_value(0)

        def sine_wave_L(x):
            return amplitude * np.sin(
                2 * np.pi * (1 / wavelengths[1]) *
                4 * (x + 0.5 * t_tracker.get_value())
            )

        # Wave always redraws (behind the letter)
        wave1_L = always_redraw(
            lambda: FunctionGraph(
                sine_wave_L, x_range=[1, 2.65, 0.01], color=BLACK
            ).shift(UP)
        )
        wave2_L = always_redraw(
            lambda: FunctionGraph(
                sine_wave_L, x_range=[3.35, 5, 0.01], color=BLACK
            ).shift(UP)
        )

        def sine_wave_S(x):
            return amplitude * np.sin(
                2 * np.pi * (1 / wavelengths[2]) *
                4 * (x + 0.5 * t_tracker.get_value())
            )

        # Wave always redraws (behind the letter)
        wave1_S = always_redraw(
            lambda: FunctionGraph(
                sine_wave_S, x_range=[1, 2.65, 0.01], color=BLACK
            ).shift(DOWN)
        )
        wave2_S = always_redraw(
            lambda: FunctionGraph(
                sine_wave_S, x_range=[3.35, 5, 0.01], color=BLACK
            ).shift(DOWN)
        )

        self.play(
            *[
                label.animate.set_color(FADE_OUT_COLOR)
                .set_fill(opacity=FADE_OUT_OPACITY)
                .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY)
                for label in labels[:-4] + labels[-1:]
            ],
            *[
                label.animate.set_color(BLACK)
                .set_fill(opacity=1)
                .set_stroke(color=BLACK, opacity=1)
                for label in labels[-3:-1]
            ],
            *[
                char_obj.animate.set_color(FADE_OUT_COLOR)
                .set_fill(opacity=FADE_OUT_OPACITY)
                .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY)
                for char_obj in char_objs[:-3] + char_objs[-1:]
            ],
            *[
                char_obj.animate.set_color(BLACK)
                .set_fill(opacity=1)
                .set_stroke(color=BLACK, opacity=1)
                for char_obj in char_objs[-3:-1]
            ],
            FadeIn(text_L, text_S),
            FadeIn(wave1_L, wave2_L, wave1_S, wave2_S),
            run_time=1,
        )

        self.play(t_tracker.animate.set_value(5),
                  run_time=3,
                  rate_func=linear)
        self.play(
            FadeOut(wave1_L, wave2_L),
            t_tracker.animate.set_value(5 + 5 / 3),
            run_time=1,
            rate_func=linear,
        )
        self.play(
            Transform(text_L, text_L_transform),
            t_tracker.animate.set_value(5 + 2 * (5 / 3)),
            run_time=1,
            rate_func=linear,
        )
        self.play(FadeOut(wave1_S, wave2_S))
        self.play(Transform(text_S, text_S_transform), run_time=1)
        self.play(FadeOut(text_S, text_L))

        # Fade out all but X
        text_X = Text("X", font_size=FONT_SIZE).set_color(
            BLACK).move_to(3 * RIGHT)
        cross = Crosshair().scale(0.75).shift(3 * RIGHT)

        t_tracker.set_value(0)

        def sine_wave_X(x):
            return amplitude * np.sin(
                2 * np.pi * (1 / wavelengths[4]) *
                4 * (x + 0.5 * t_tracker.get_value())
            )

        # Wave always redraws (behind the letter)
        wave1_X = always_redraw(
            lambda: FunctionGraph(sine_wave_X,
                                  x_range=[1, 2.65, 0.01],
                                  color=BLACK)
        )

        wave2_X = always_redraw(
            lambda: FunctionGraph(sine_wave_X,
                                  x_range=[3.35, 5, 0.01],
                                  color=BLACK)
        )

        self.play(
            FadeIn(text_X),
            FadeIn(wave1_X, wave2_X),
            *[
                label.animate.set_color(FADE_OUT_COLOR)
                .set_fill(opacity=FADE_OUT_OPACITY)
                .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY)
                for label in labels[:-6] + labels[-4:]
            ],
            *[
                label.animate.set_color(BLACK)
                .set_fill(opacity=1)
                .set_stroke(color=BLACK, opacity=1)
                for label in labels[-5]
            ],
            *[
                char_obj.animate.set_color(FADE_OUT_COLOR)
                .set_fill(opacity=FADE_OUT_OPACITY)
                .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY)
                for char_obj in char_objs[:-6] + char_objs[-4:]
            ],
            *[
                char_obj.animate.set_color(BLACK)
                .set_fill(opacity=1)
                .set_stroke(color=BLACK, opacity=1)
                for char_obj in char_objs[-5]
            ],
            run_time=1,
        )

        self.play(t_tracker.animate.set_value(5), run_time=3, rate_func=linear)
        self.play(FadeOut(wave1_X, wave2_X))
        self.play(FadeOut(text_X), FadeIn(cross))
        self.play(cross.animate.shift(1.5 * UP + 1.5 * RIGHT).scale(0.75))
        self.play(cross.animate.shift(2.5 * DOWN + 3 * LEFT).scale(1.3))
        self.play(cross.animate.shift(
            1 * UP + 1.5 * RIGHT).scale((1 / 0.75) / 1.3))
        self.play(FadeOut(cross))

        text_X = Text("X", font_size=FONT_SIZE).set_color(
            BLACK).move_to(3 * RIGHT + UP)
        text_S = (
            Text("S", font_size=FONT_SIZE).set_color(
                BLACK).move_to(3 * RIGHT + DOWN)
        )

        # Fade out all but C
        text_C = Text("C", font_size=FONT_SIZE).set_color(
            BLACK).move_to(3 * RIGHT)
        t_tracker.set_value(0)

        def sine_wave_C(x):
            return amplitude * np.sin(
                2 * np.pi * (1 / wavelengths[3]) *
                4 * (x + 0.5 * t_tracker.get_value())
            )

        # Wave always redraws (behind the letter)
        wave1_C = always_redraw(
            lambda: FunctionGraph(sine_wave_C, x_range=[
                                  1, 2.65, 0.01], color=BLACK)
        )

        wave2_C = always_redraw(
            lambda: FunctionGraph(sine_wave_C, x_range=[
                                  3.35, 5, 0.01], color=BLACK)
        )

        shift_tracker_X = ValueTracker(1)
        # Wave always redraws (behind the letter)
        wave1_X = always_redraw(
            lambda: FunctionGraph(
                sine_wave_X, x_range=[1, 2.65, 0.01], color=BLACK
            ).shift(UP * shift_tracker_X.get_value())
        )

        wave2_X = always_redraw(
            lambda: FunctionGraph(
                sine_wave_X, x_range=[3.35, 5, 0.01], color=BLACK
            ).shift(UP * shift_tracker_X.get_value())
        )

        shift_tracker_S = ValueTracker(1)
        # Wave always redraws (behind the letter)
        wave1_S = always_redraw(
            lambda: FunctionGraph(
                sine_wave_S, x_range=[1, 2.7, 0.01], color=BLACK
            ).shift(DOWN * shift_tracker_S.get_value())
        )

        wave2_S = always_redraw(
            lambda: FunctionGraph(
                sine_wave_S, x_range=[3.3, 5, 0.01], color=BLACK
            ).shift(DOWN * shift_tracker_S.get_value())
        )

        self.play(FadeIn(text_X, text_S, wave1_S, wave2_S, wave1_X, wave2_X))
        self.play(t_tracker.animate.set_value(5), run_time=3, rate_func=linear)

        self.play(
            FadeIn(text_C),
            FadeIn(wave1_C, wave2_C),
            text_X.animate.shift(1 * UP),
            text_S.animate.shift(1 * DOWN),
            shift_tracker_X.animate.set_value(2),
            shift_tracker_S.animate.set_value(2),
            t_tracker.animate.set_value(5 + 5 / 3),
            *[
                label.animate.set_color(FADE_OUT_COLOR)
                .set_fill(opacity=FADE_OUT_OPACITY)
                .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY)
                for label in labels[:-5] + labels[-5:]
            ],
            *[
                label.animate.set_color(BLACK)
                .set_fill(opacity=1)
                .set_stroke(color=BLACK, opacity=1)
                for label in labels[-4]
            ],
            *[
                char_obj.animate.set_color(FADE_OUT_COLOR)
                .set_fill(opacity=FADE_OUT_OPACITY)
                .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY)
                for char_obj in char_objs[:-5] + char_objs[-5:]
            ],
            *[
                char_obj.animate.set_color(BLACK)
                .set_fill(opacity=1)
                .set_stroke(color=BLACK, opacity=1)
                for char_obj in char_objs[-4]
            ],
            run_time=1,
            rate_func=linear,
        )

        self.play(
            t_tracker.animate.set_value(5 + 5 + 5 / 3), run_time=3, rate_func=linear
        )
        self.play(
            FadeOut(wave1_C, wave2_C),
            t_tracker.animate.set_value(5 + 5 + 2 * 5 / 3),
            run_time=1,
            rate_func=linear,
        )

        text_C_transform = (
            Text("Compromise", font_size=FONT_SIZE).set_color(
                BLACK).move_to(3 * RIGHT)
        )

        self.play(
            Transform(text_C, text_C_transform),
            t_tracker.animate.set_value(5 + 5 + 3 * (5 / 3)),
            run_time=1,
            rate_func=linear,
        )
        self.play(
            t_tracker.animate.set_value(5 + 5 + 4 * (5 / 3)),
            run_time=1,
            rate_func=linear,
        )
        self.play(FadeOut(text_S, text_C, text_X,
                  wave1_S, wave2_S, wave1_X, wave2_X))

        t_tracker.set_value(0)

        def sine_wave_K(x):
            return amplitude * np.sin(
                2 * np.pi * (1 / wavelengths[6]) *
                4 * (x + 0.5 * t_tracker.get_value())
            )

        # Wave always redraws (behind the letter)
        wave1_K = always_redraw(
            lambda: FunctionGraph(sine_wave_K, x_range=[
                                  1, 2.6, 0.01], color=BLACK)
        )

        wave2_K = always_redraw(
            lambda: FunctionGraph(sine_wave_K, x_range=[
                                  3.35, 5, 0.01], color=BLACK)
        )

        def sine_wave_Ka(x):
            return amplitude * np.sin(
                2 * np.pi * (1 / wavelengths[5]) *
                4 * (x + 0.5 * t_tracker.get_value())
            )

        # Wave always redraws (behind the letter)
        wave1_Ka = always_redraw(
            lambda: FunctionGraph(
                sine_wave_Ka, x_range=[1, 2.4, 0.01], color=BLACK
            ).shift(1.5 * UP)
        )

        wave2_Ka = always_redraw(
            lambda: FunctionGraph(
                sine_wave_Ka, x_range=[3.6, 5, 0.01], color=BLACK
            ).shift(1.5 * UP)
        )

        def sine_wave_Ku(x):
            return amplitude * np.sin(
                2 * np.pi * (1 / wavelengths[7]) *
                4 * (x + 0.5 * t_tracker.get_value())
            )

        # Wave always redraws (behind the letter)
        wave1_Ku = always_redraw(
            lambda: FunctionGraph(
                sine_wave_Ku, x_range=[1, 2.4, 0.01], color=BLACK
            ).shift(1.5 * DOWN)
        )

        wave2_Ku = always_redraw(
            lambda: FunctionGraph(
                sine_wave_Ku, x_range=[3.6, 5, 0.01], color=BLACK
            ).shift(1.5 * DOWN)
        )

        text_K = Text("K", font_size=FONT_SIZE).set_color(
            BLACK).move_to(3 * RIGHT)
        text_kurz = (
            Text("Kurz", font_size=FONT_SIZE).set_color(
                BLACK).move_to(3 * RIGHT)
        )
        text_K_old = Text("K", font_size=FONT_SIZE).set_color(
            BLACK).move_to(3 * RIGHT)

        text_Ka = (
            Text("Ka", font_size=FONT_SIZE)
            .set_color(BLACK)
            .move_to(3 * RIGHT)
            .shift(1.5 * UP)
        )
        text_Ka_t = (
            Text("K-above", font_size=FONT_SIZE)
            .set_color(BLACK)
            .move_to(3 * RIGHT)
            .shift(1.5 * UP)
        )

        text_Ku = (
            Text("Ku", font_size=FONT_SIZE)
            .set_color(BLACK)
            .move_to(3 * RIGHT)
            .shift(1.5 * DOWN)
        )
        text_Ku_t = (
            Text("K-under", font_size=FONT_SIZE)
            .set_color(BLACK)
            .move_to(3 * RIGHT)
            .shift(1.5 * DOWN)
        )
        text_Ku_old = (
            Text("Ku", font_size=FONT_SIZE)
            .set_color(BLACK)
            .move_to(3 * RIGHT)
            .shift(1.5 * DOWN)
        )

        # Fade out all but Ku, K, Ka
        self.play(
            FadeIn(text_K),
            FadeIn(wave1_K, wave2_K),
            *[
                label.animate.set_color(FADE_OUT_COLOR)
                .set_fill(opacity=FADE_OUT_OPACITY)
                .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY)
                for label in labels[-5:]
            ],
            *[
                label.animate.set_color(BLACK)
                .set_fill(opacity=1)
                .set_stroke(color=BLACK, opacity=1)
                for label in labels[:-5]
            ],
            *[
                char_obj.animate.set_color(FADE_OUT_COLOR)
                .set_fill(opacity=FADE_OUT_OPACITY)
                .set_stroke(color=FADE_OUT_COLOR, opacity=FADE_OUT_OPACITY)
                for char_obj in char_objs[-5:]
            ],
            *[
                char_obj.animate.set_color(BLACK)
                .set_fill(opacity=1)
                .set_stroke(color=BLACK, opacity=1)
                for char_obj in char_objs[:-5]
            ],
            run_time=1,
        )
        self.play(t_tracker.animate.set_value(5), run_time=3, rate_func=linear)
        self.play(FadeOut(wave1_K, wave2_K))
        self.play(Transform(text_K, text_kurz))
        self.wait(2)
        self.play(Transform(text_K, text_K_old))
        self.play(FadeIn(wave1_K, wave2_K, wave1_Ku, wave2_Ku, text_Ku))

        self.play(t_tracker.animate.set_value(5 + 5),
                  run_time=3,
                  rate_func=linear)
        self.play(
            FadeOut(wave1_Ku, wave2_Ku),
            t_tracker.animate.set_value(5 + 1 * 5 / 3 + 5),
            run_time=1,
            rate_func=linear,
        )
        self.play(
            Transform(text_Ku, text_Ku_t),
            t_tracker.animate.set_value(5 + 2 * 5 / 3 + 5),
            run_time=1,
            rate_func=linear,
        )
        self.play(
            Transform(text_Ku, text_Ku_t),
            t_tracker.animate.set_value(5 + 3 * 5 / 3 + 5),
            run_time=1,
            rate_func=linear,
        )
        self.play(
            t_tracker.animate.set_value(5 + 4 * 5 / 3 + 5),
            run_time=1,
            rate_func=linear
        )
        self.play(
            Transform(text_Ku, text_Ku_old),
            t_tracker.animate.set_value(5 + 5 * 5 / 3 + 5),
            run_time=1,
            rate_func=linear,
        )

        self.play(
            FadeIn(text_Ka, wave1_Ka, wave2_Ka, wave1_Ku, wave2_Ku),
            t_tracker.animate.set_value(5 + 5 + 6 * (5 / 3)),
            run_time=1,
            rate_func=linear,
        )
        self.play(
            t_tracker.animate.set_value(5 + 5 + 5 + 6 * (5 / 3)),
            run_time=3,
            rate_func=linear,
        )
        self.play(
            FadeOut(wave1_Ka, wave2_Ka),
            t_tracker.animate.set_value(5 + 5 + 5 + 7 * (5 / 3)),
            run_time=1,
            rate_func=linear,
        )
        self.play(
            Transform(text_Ka, text_Ka_t),
            t_tracker.animate.set_value(5 + 5 + 5 + 8 * (5 / 3)),
            run_time=1,
            rate_func=linear,
        )
        self.play(
            t_tracker.animate.set_value(5 + 5 + 5 + 5 + 8 * (5 / 3)),
            run_time=3,
            rate_func=linear,
        )

        self.play(
            FadeOut(
                *labels,
                *char_objs,
                table_line,
                bands_title,
                lambda_title,
                text_Ku,
                text_Ka,
                text_K,
                wave1_K,
                wave2_K,
                wave1_Ku,
                wave2_Ku,
            )
        )

    def construct(self):

        self.camera.background_color = WHITE

        fw = config.frame_width  # default 14
        fh = config.frame_height  # default 8

        # SAR flying
        self.play_scene_1(fw, fh)

        # Wavelengths illustration and interval table
        self.play_scene_2()

        # P & L band intro
        self.play_scene_3()

        # P & L band applications
        self.play_scene_4(fw, fh)

        # S, C, X intro
        self.play_scene_5()

        # S, C, X band application
        self.play_scene_6(fw, fh)

        # Ku, K, Ka band intro
        self.play_scene_7()

        # Ku, K, Ka band application
        # Cryosat also uses spotlight-similar imaging
        # see "Evaluation of SAR altimetry over the antarctic
        # ice sheet from CryoSat-2 acquisitions"
        # Aublanc et al. 2018
        self.play_scene_8(fw, fh)

        # comparison S, L, P in F-SAR
        self.play_scene_9()

        # explanation of names of bands
        self.play_scene_10()
