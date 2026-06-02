"""Outro (Resources, Notebook, Code)"""

from manim import *
from utils.colors import *

FILES_DIR = "./02_bridges_in_stripmap/pngs/"
PLACEHOLDER_PATH = f"{FILES_DIR}/placeholder.png"


class Scene7(MovingCameraScene):
    def construct(self):
        self.camera.background_color = "#e6d8bc"
        fw = config.frame_width  # default 14
        fh = config.frame_height  # default 8

        axes = Axes(
            x_range=[-fw / 2, fw / 2, 1],
            y_range=[-fh / 2, fh / 2, 1],
            x_length=fw,
            y_length=fh,
            tips=False,
        ).set_color(GREY)

        axes.add_coordinates(color=GREY)
        # self.add(axes)
        # 0:00-0:03 -> like and subscribe
        logo = ImageMobject(PLACEHOLDER_PATH)
        logo.scale(0.3)
        self.play(FadeIn(logo), run_time=1)
        self.wait(2)

        # 0:03-0:06 -> description for resources
        desc_text = Text(
            "DESCRIPTION",
            font_size=DEFAULT_FONT_SIZE,
            font="Zalando Sans SemiExpanded",
            color=TEXT_COLOR,
        )
        desc_text.shift(4 * DOWN)

        resources_text = Text(
            "resources",
            font_size=DEFAULT_FONT_SIZE / 1.5,
            font="Zalando Sans",
            color=TEXT_COLOR,
        )
        resources_text.next_to(desc_text, LEFT, buff=1).shift(DOWN * 2.5)

        curved_arrow = CurvedArrow(
            desc_text.get_bottom() + LEFT * 0.1,
            resources_text.get_top() + UP * 0.1,
            angle=-PI / 2,
            color=TEXT_COLOR,
        )

        curved_arrow = CubicBezier(
            desc_text.get_left() + LEFT * 0.1,  # start
            desc_text.get_left() + LEFT * 1.5,  # control point 1
            resources_text.get_top() + UP * 1.5,  # control point 2
            resources_text.get_top() + UP * 0.1,  # end
            color=COLOR_SB,
        )
        self.play(
            FadeIn(desc_text), self.camera.frame.animate.shift(DOWN * 5.6), run_time=1
        )
        self.camera.frame.save_state()
        self.play(Create(curved_arrow), run_time=0.5)
        self.play(FadeIn(resources_text), run_time=0.5)
        self.wait(1)

        # 0:06-0:09 -> CPHD notebook
        notebook = ImageMobject(PLACEHOLDER_PATH)
        notebook.scale(0.4).next_to(desc_text, DOWN, buff=0.1).shift(DOWN * 1)

        notebook_rect = SurroundingRectangle(
            notebook, color=BLACK, buff=0, stroke_width=1
        )

        notebook_arrow = CubicBezier(
            desc_text.get_bottom() + DOWN * 0.1,
            desc_text.get_bottom() + DOWN * 1.5,
            notebook.get_top() + UP * 1.5,
            notebook.get_top() + UP * 0.1,
            color=COLOR_TB,
        )
        self.play(
            Create(notebook_arrow),
            FadeIn(notebook),
            FadeIn(notebook_rect),
            run_time=0.5,
        )
        self.play(self.camera.frame.animate.scale(0.45).move_to(notebook), run_time=1)
        self.wait(0.5)
        self.play(Restore(self.camera.frame), run_time=1)

        # 0:09-0:12 -> source code video
        github_text = Text(
            "github/sigmanought",
            font_size=DEFAULT_FONT_SIZE / 1.5,
            font="Zalando Sans",
            color=TEXT_COLOR,
        )
        github_text.next_to(desc_text, RIGHT, buff=0.1).shift(DOWN * 2.5)

        github_arrow = CubicBezier(
            desc_text.get_right() + RIGHT * 0.2,
            desc_text.get_right() + RIGHT * 1.5,
            github_text.get_top() + UP * 1.5,
            github_text.get_top() + UP * 0.2,
            color=COLOR_DB,
        )

        self.play(Create(github_arrow), run_time=0.7)
        self.play(FadeIn(github_text), run_time=0.8)
        self.wait(1)

        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1,
        )
