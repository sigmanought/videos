"""Utilities for animations."""

from manim import *


def draw_angle(line1, line2, radius, color=GREEN, opposite_angle=False):
    # Compute the start and end angles for the sector
    # In Manim, angles are measured counterclockwise from the x-axis
    start_vector = line1.get_start() - line1.get_end()
    end_vector = line2.get_start() - line2.get_end()

    # Compute angles relative to x-axis
    start_angle = np.arctan2(start_vector[1], start_vector[0])
    end_angle = np.arctan2(end_vector[1], end_vector[0])
    if opposite_angle:
        start_angle += np.pi
        end_angle += np.pi

    # Create a filled sector
    wedge = Sector(
        start_angle=start_angle,
        angle=end_angle - start_angle,
        radius=radius,
        fill_color=color,
        fill_opacity=0.5,
        stroke_width=0,
    ).move_arc_center_to(
        line1.get_end()
    )  # move to vertex

    return wedge
