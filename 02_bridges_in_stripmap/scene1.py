"""Comparison of optical and SAR image of the Golden Gate bridge."""

from pathlib import Path

from manim import *
from utils.colors import *
from utils.decoration import create_shadow

fast = 1
RES_TYPE = "low"
IMG_WIDTH = 6
IMG_HEIGHT = 4

# Directory which contains the images. Due to licensing
# only a placeholder is included. The images can be downloaded
# from the Capella Space SAR Open Dataset on AWS
# and from the PlanetLabs Open Data on AWS.
FILES_DIR = "./02_bridges_in_stripmap/pngs/"
OPTICAL_IMG_PATH = f"{FILES_DIR}/placeholder.png"
OPTICAL_IMG_DOWNSAMPLED_PATH = f"{FILES_DIR}/placeholder.png"
OPTICAL_IMG_CLOUDS_PATH = f"{FILES_DIR}/placeholder.png"
SAR_IMAGE = f"{FILES_DIR}/placeholder.png"

# directory with frames of the three videos to be
# previewed at the end of Scene 1
VIDEO_FRAMES_DIR = "..."
PLAY_PREVIEW = False


class Scene1(MovingCameraScene):
    def construct(self):
        self.camera.background_color = "#e6d8bc"

        # Load images
        optical_image = ImageMobject(OPTICAL_IMG_PATH).scale(0.07)
        optical_image_ds = ImageMobject(OPTICAL_IMG_DOWNSAMPLED_PATH).scale(0.07)
        optical_image_clouds = ImageMobject(OPTICAL_IMG_CLOUDS_PATH).scale(0.07)

        sar_image = ImageMobject(SAR_IMAGE).scale(0.07)

        optical_image.stretch_to_fit_width(IMG_WIDTH)
        optical_image.stretch_to_fit_height(IMG_HEIGHT)

        optical_image_ds.stretch_to_fit_width(IMG_WIDTH)
        optical_image_ds.stretch_to_fit_height(IMG_HEIGHT)

        optical_image_clouds.stretch_to_fit_width(IMG_WIDTH)
        optical_image_clouds.stretch_to_fit_height(IMG_HEIGHT)

        sar_image.stretch_to_fit_width(IMG_WIDTH)
        sar_image.stretch_to_fit_height(IMG_HEIGHT)

        # Move to center
        optical_image.to_edge(LEFT).shift(RIGHT * 0.2 + 0.25 * DOWN)
        optical_image_ds.move_to(optical_image.get_center())
        optical_image_clouds.move_to(optical_image.get_center())

        sar_image.to_edge(RIGHT).shift(LEFT * 0.2 + 0.25 * DOWN)

        # Add titles and sources
        optical_label = (
            Text("Optical", font="Zalando Sans SemiExpanded")
            .next_to(optical_image, UP)
            .set_color(TEXT_COLOR)
            .set_stroke(TEXT_COLOR)
        )
        sar_label = (
            Text("SAR", font="Zalando Sans SemiExpanded")
            .next_to(sar_image, UP)
            .set_color(TEXT_COLOR)
            .set_stroke(TEXT_COLOR)
            .shift(0.12 * UP)
        )

        # add two transparent letters so that all words
        # are on one line and the phrase is centered
        scale_text = 0.75
        synthetic = (
            Text("ySynthetic", font="Zalando Sans SemiExpanded")
            .set_color(TEXT_COLOR)
            .scale(scale_text)
        )
        synthetic[0].set_opacity(0)
        radar = (
            Text("Radary", font="Zalando Sans SemiExpanded")
            .set_color(TEXT_COLOR)
            .scale(scale_text)
        )  # the "y" is here to center the text height
        radar[5].set_opacity(0)  # make y transparent

        sar_label_full = (
            VGroup(
                Text("Synthetic", font="Zalando Sans SemiExpanded")
                .set_color(TEXT_COLOR)
                .scale(scale_text),
                Text("Aperture", font="Zalando Sans SemiExpanded")
                .set_color(TEXT_COLOR)
                .scale(scale_text),
                radar,
            )
            .arrange(RIGHT, buff=0.23)
            .move_to(sar_label.get_center())
            .shift(0.1 * RIGHT)
        )

        optical_source = (
            Text(
                "Source: Planet Labs, CC BY 4.0\n(creativecommons.org/licenses/by/4.0/)",
                font_size=SOURCE_FONT_SIZE,
                font="Zalando Sans",
            )
            .set_color(SOURCE_COLOR)
            .scale(SOURCE_SCALE)
        )
        sar_source = (
            Text(
                "Source: Capella Space, CC BY 4.0\n(creativecommons.org/licenses/by/4.0/)",
                font_size=SOURCE_FONT_SIZE,
                font="Zalando Sans",
            )
            .set_color(SOURCE_COLOR)
            .scale(SOURCE_SCALE)
        )

        sar_source.next_to(sar_image, DOWN).align_to(sar_image, LEFT)
        optical_source.next_to(optical_image, DOWN).align_to(optical_image, LEFT)

        # Create a slightly bigger black rectangle as shadow/backdrop
        optical_shadow = create_shadow(
            optical_image, layers=20, scale_factor=1.1, max_opacity=0.1
        )
        sar_shadow = create_shadow(
            sar_image, layers=20, scale_factor=1.1, max_opacity=0.1
        )

        self.camera.frame.shift(0.25 * DOWN)
        self.play(
            FadeIn(
                optical_shadow,
                optical_image,
                sar_shadow,
                sar_image,
                optical_source,
                sar_source,
                run_time=fast * 2,
            )
        )
        self.wait(fast * 2)

        scale_fac = 1.05
        optical_shadow.set_z_index(0)
        optical_image.set_z_index(1)
        sar_shadow.set_z_index(0)
        sar_image.set_z_index(1)

        self.play(
            Succession(
                AnimationGroup(
                    ScaleInPlace(optical_image, scale_fac),
                    ScaleInPlace(optical_shadow, scale_fac),
                ),
                AnimationGroup(
                    ScaleInPlace(optical_image, 1 / scale_fac),
                    ScaleInPlace(optical_shadow, 1 / scale_fac),
                ),
                run_time=fast * 1,
            )
        )
        self.play(
            FadeIn(optical_label),
            self.camera.frame.animate.shift(0.25 * UP),
            run_time=fast * 1,
        )

        self.play(
            Succession(
                AnimationGroup(
                    ScaleInPlace(sar_image, scale_fac),
                    ScaleInPlace(sar_shadow, scale_fac),
                ),
                AnimationGroup(
                    ScaleInPlace(sar_image, 1 / scale_fac),
                    ScaleInPlace(sar_shadow, 1 / scale_fac),
                ),
                run_time=fast * 1,
            )
        )
        self.play(FadeIn(sar_label_full), run_time=fast * 1)

        self.wait(fast * 1)

        for word in sar_label_full:
            self.play(
                Indicate(word, scale_factor=1.1, color=BLACK, run_time=fast * 0.3)
            )
            self.wait(fast * 0.1)
        self.wait(0.5)
        self.play(ReplacementTransform(sar_label_full, sar_label), run_time=fast * 0.8)
        self.wait(fast * 1)

        self.play(
            FadeOut(
                optical_image, optical_shadow, optical_label, optical_source, sar_label
            ),
            run_time=fast * 1,
        )
        sar_group = Group(sar_shadow, sar_image, sar_source)
        shift_vector = ORIGIN - sar_image.get_center()
        offset_y = sar_source.get_center()[1] - sar_image.get_center()[1]
        self.play(sar_group.animate.shift(shift_vector))
        offset_y = sar_source.get_center()[1] - sar_image.get_center()[1]
        new_left_edge = sar_image.get_left()[0] * 1.4
        offset_x = sar_source.get_center()[0] - sar_image.get_left()[0]

        self.play(
            sar_image.animate.scale(1.4),
            sar_shadow.animate.scale(1.4),
            sar_source.animate.move_to(
                [
                    new_left_edge + offset_x,  # Keep same offset from left edges
                    ORIGIN[1] + offset_y * 1.4,  # Scale vertical offset proportionally
                    0,
                ]
            ),
            run_time=fast * 1,
        )
        self.wait(1)
        # Define bounce types for reading images
        bounce_types = ["single", "double", "triple", "all"]

        # Dictionary to store all images
        highlighted_images = {}

        for bounce_type in bounce_types:
            img = ImageMobject(SAR_IMAGE).scale(0.07)
            img.stretch_to_fit_width(IMG_WIDTH)
            img.stretch_to_fit_height(IMG_HEIGHT)
            img.scale(1.4).move_to(sar_image.get_center())
            img.set_z_index(2)
            highlighted_images[bounce_type] = img

        self.wait(fast * 0.5)
        for bounce_type in bounce_types:
            self.play(FadeIn(highlighted_images[bounce_type]), run_time=fast * 0.3)
            self.wait(fast * 0.5)
            if not bounce_type == "all":
                pass
            else:
                self.wait(2)
        self.remove(
            *[highlighted_images[bounce_type] for bounce_type in bounce_types[:-1]],
            sar_image,
        )
        sar_group = Group(sar_shadow, img, sar_source)
        self.play(sar_group.animate.scale(1 / 1.4).shift(3 * LEFT))

        phase_tracker = ValueTracker(0)

        colors = [
            "#4D80FF",
            "#009900",
            "#FFA600",
        ]
        offsets = [1.2, 0, -1.2]
        phases = [0, 0, 0]

        waves = VGroup(
            *[
                always_redraw(
                    lambda y=offset, c=color, p=phase: ParametricFunction(
                        lambda t: np.array(
                            [
                                t - 3,
                                0.18 * np.sin(4 * t + phase_tracker.get_value() + p)
                                + y,
                                0,
                            ]
                        ),
                        t_range=[4.5, 8.5],
                        color=c,
                        stroke_width=5,
                    )
                )
                for offset, color, phase in zip(offsets, colors, phases)
            ]
        )

        self.play(FadeIn(*waves), run_time=0.5)
        # run for 5 seconds
        self.play(
            phase_tracker.animate.set_value(3 * TAU * 2.5),
            run_time=5,
            rate_func=linear,
        )

        self.remove(
            sar_image,
            highlighted_images["single"],
            highlighted_images["double"],
            highlighted_images["triple"],
        )

        self.play(FadeOut(highlighted_images["all"], sar_source, sar_shadow, *waves))

        if PLAY_PREVIEW:
            # make 3 panels, each gets a video.
            r1 = Rectangle(width=4, height=2.4).set_stroke(BLACK, width=2)
            r2 = Rectangle(width=4, height=2.4).set_stroke(BLACK, width=2)
            r3 = Rectangle(width=4, height=2.4).set_stroke(BLACK, width=2)

            # Arrange into 1x3 grid
            panel = VGroup(r1, r2, r3).arrange(RIGHT, buff=0.5)

            def make_video(folder, rect, start=0, end=None, fps=60, speed=1):
                """Play a video from .png files within the video."""
                frames = sorted(Path(folder).glob("*.png"))

                if end is None:
                    end = len(frames)

                frames = frames[start:end]

                img = ImageMobject(str(frames[0]))
                img.replace(rect)
                img.move_to(rect)

                state = {"i": 0, "t": 0}

                def updater(mob, dt):
                    state["t"] += dt
                    if state["t"] >= 1 / fps:
                        state["t"] = 0
                        state["i"] = int(state["i"] + speed) % len(frames)

                        if state["i"] >= len(frames):
                            state["i"] = 0

                        mob.become(ImageMobject(str(frames[state["i"]])))
                        mob.replace(rect)
                        mob.move_to(rect)

                img.add_updater(updater)
                return img

            # create videos, these sequences of images
            v1 = make_video(
                f"{VIDEO_FRAMES_DIR}/1/",
                r1,
                speed=1.5,
            )

            v2 = make_video(
                f"{VIDEO_FRAMES_DIR}/2/",
                r2,
                start=1432,
                end=2395,
                speed=1.5,
            )

            v3 = make_video(
                f"{VIDEO_FRAMES_DIR}/3/",
                r3,
                start=1530,
                end=4299,
                speed=3.5,
            )
            self.add(v1, v2, v3)

            self.play(FadeIn(panel))
            frame = self.camera.frame
            panels = [r1, r2, r3]
            wait_times = [4, 4, 6]

            # zoom into each panel
            for p, wait_time in zip(panels, wait_times):
                self.play(frame.animate.move_to(p).set(width=p.width * 1.5))
                self.wait(wait_time)

            # zoom back out
            self.play(frame.animate.move_to(ORIGIN).set(width=14))
            self.wait(2)
