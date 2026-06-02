"""Utilities for decoration, such as backdrops of images."""

from manim import *


def create_shadow(
    vmobject,
    layers=10,
    scale_factor=1.3,
    color=BLACK,
    max_opacity=0.3,
    corner_radius=0.3,
    slant=None,
    rotate=None,
):
    """
    Creates a subtle layered shadow behind a VMobject (like an ImageMobject)."""
    shadow_group = VGroup()
    for i in range(layers):
        opacity = max_opacity * (1 - i / layers)
        if slant is not None:
            width = vmobject.width * (1 + (scale_factor - 1) * (i / layers))
            height = vmobject.height * (1 + (scale_factor - 1) * (i / layers))
            bottom_left = np.array([0, 0, 0])
            bottom_right = np.array([width, 0, 0])
            top_right = np.array([width + slant, height, 0])
            top_left = np.array([slant, height, 0])

            rect = Polygon(
                bottom_left,
                bottom_right,
                top_right,
                top_left,
                color=color,
                fill_opacity=opacity,
                stroke_opacity=0,
            )
            rect.round_corners(radius=0.3)

        else:
            rect = RoundedRectangle(
                width=vmobject.width * (1 + (scale_factor - 1) * (i / layers)),
                height=vmobject.height * (1 + (scale_factor - 1) * (i / layers)),
                fill_color=color,
                fill_opacity=opacity,
                stroke_opacity=0,
                corner_radius=corner_radius,
            )
        if rotate is not None:
            rect.rotate(rotate * DEGREES)  # counterclockwise
        rect.move_to(vmobject.get_center())
        shadow_group.add(rect)
    return shadow_group
