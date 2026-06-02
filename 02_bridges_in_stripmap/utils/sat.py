"""Satellite Moving Objects."""

import numpy as np
from manim import *
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup, OpenGLVMobject
from PIL import Image

LIGHT_BEIGE = ManimColor("#e8dfcc")


def create_scalloped_cap(
    radius=20.0, height=2.0, num_scallops=6, scallop_depth=1, mesh_res=[40, 40]
):
    """
    Spherical cap with scalloped cutouts carved from the bottom,
    while preserving a perfect spherical surface.
    """

    # Fixed cap angle
    base_z_min = radius - height
    phi_cap = np.arccos(base_z_min / radius)

    def scalloped_phi_cut(v):
        angle_step = TAU / num_scallops
        t = ((v + angle_step / 2) % angle_step) / angle_step

        # triangular wave → sharp peaks
        sharp = abs(np.sin(t * TAU))

        z_cut = base_z_min + scallop_depth * sharp
        return np.arccos(z_cut / radius)

    def rim_point(v):
        # 3D point at u=1 for a given angle v
        phi = scalloped_phi_cut(v)
        return radius * np.array(
            [np.sin(phi) * np.cos(v), np.sin(phi) * np.sin(v), np.cos(phi)]
        )

    cap = Surface(
        lambda u, v: (
            radius
            * np.array(
                [
                    np.sin(min(u * phi_cap, scalloped_phi_cut(v))) * np.cos(v),
                    np.sin(min(u * phi_cap, scalloped_phi_cut(v))) * np.sin(v),
                    np.cos(min(u * phi_cap, scalloped_phi_cut(v))),
                ]
            )
        ),
        u_range=[0, 1],
        v_range=[0, TAU],
        resolution=(
            mesh_res[0],
            num_scallops * 10,
        ),
    )

    return cap, rim_point


def cylinder_between_points(p1, p2, radius=0.05, color=None):
    # Vector from p1 to p2
    axis = p2 - p1
    height = np.linalg.norm(axis)
    if height == 0:
        return None  # points coincide

    # Unit vector along axis
    axis_normalized = axis / height

    # Default cylinder is along UP (z-axis)
    default_axis = np.array([0, 0, 1])

    # Compute rotation axis and angle
    rot_axis = np.cross(default_axis, axis_normalized)
    rot_axis_norm = np.linalg.norm(rot_axis)

    if rot_axis_norm < 1e-6:
        rot_axis = UP  # arbitrary axis, no rotation needed
        rot_angle = 0
    else:
        rot_axis = rot_axis / rot_axis_norm
        rot_angle = np.arccos(np.clip(np.dot(default_axis, axis_normalized), -1, 1))

    # Create cylinder along z-axis
    cyl = Cylinder(radius=radius, height=height, color=color)
    cyl.move_to((p1 + p2) / 2)  # move to midpoint
    if rot_angle != 0:
        cyl.rotate(rot_angle, axis=rot_axis, about_point=(p1 + p2) / 2)

    return cyl


# cylinders that connect the mesh and net
def cylinder_on_mesh(p1, p2, radius=0.1, color=BLUE):
    axis = p2 - p1
    height = np.linalg.norm(axis)
    direction = axis / height

    cyl = Cylinder(radius=radius, height=height, direction=direction, color=color)

    cyl.move_to((p1 + p2) / 2)
    return cyl


def create_truss_opengl(
    num_scallops, rim1, shift_mesh, height, full_range=False, half_truss=False
):
    """Create truss that holds net above mesh."""
    cylinders = []
    # we hide the last three truss cylinders in opengl
    # otherwise they render on top of the box and solar array
    # if you want all scallops then just use range(0, num_scallops*2 )
    if full_range:
        truss_idx = range(0, num_scallops * 2)
    if half_truss:
        truss_idx = range(0, num_scallops)
    else:
        truss_idx = [0, 1, 2] + list(range(6, 2 * num_scallops))
    for i in truss_idx:
        # angle at start of scallop
        v = i * TAU / num_scallops / 2

        # rim point in local cap coordinates
        p_local1 = rim1(v)

        # convert to world coordinates if cap1 has been shifted
        p_world1 = p_local1
        p_world2 = p_local1

        # create start and end point of cylinders
        dot1 = Dot(p_world1, radius=0.2, color=RED).shift(IN * shift_mesh)
        dot2 = Dot(p_world2, radius=0.2, color=RED).shift(
            IN * (shift_mesh - 0.8) + height * OUT
        )

        cyl = cylinder_on_mesh(
            dot1.get_center(), dot2.get_center(), radius=0.03, color=BLACK
        ).set_stroke(BLACK)

        cylinders.append(cyl)
    return cylinders


def create_solar_array_opengl(
    rows=3,
    cols=2,
    rect_width=2,
    rect_height=1.5,
    rect_depth=0.02,
    horizontal_buffer=0.7,
    vertical_buffer=0.2,
    color=GREY_E,
):
    """Create a grid of rectangles as a solar array in OpenGL."""
    rectangles = OpenGLVGroup()

    for row in range(rows):
        for col in range(cols):
            rect = Cube()
            rect.stretch_to_fit_width(rect_width)  # → X axis (left/right)
            rect.stretch_to_fit_height(rect_height)  # → Y axis (up/down)
            rect.stretch(rect_depth, dim=2)  # → Z axis (in/out of screen)

            x_pos = (
                col * (rect_width + horizontal_buffer)
                - (cols - 1) * (rect_width + horizontal_buffer) / 2
            )
            y_pos = -row * (rect_height + vertical_buffer)

            rect = rect.copy()
            rect.move_to([x_pos, y_pos, 0])
            rect.set_color(color).set_fill(opacity=1)
            rectangles.add(rect)

    return rectangles


class OpenGLMeshReflectorSat3d(OpenGLVMobject):
    """Mesh Reflector SAR Satellite in 3D with OpenGL rendering.
    For OpenGL we only use the lower mesh (no net). The net has
    opacity = 0 and stroke fill = 1. Such objects will always be rendered
    last and appear on top of other objects, so net will always
    render on top of the solar array. The default z_indices
    show the satellite from the back, with the solar array and
    processing box closest to the viewer."""

    def __init__(
        self,
        height=0.5,
        num_scallops=6,
        rect_width=1.5,
        # rect_height = 2,
        radius_mesh_base=15,
        scallop_depth=0.1,
        box_z_index=2,
        cyl_z_index=100,
        solar_array_z_index=300,
        mesh_z_index=1,
        back_view=False,
        full_truss=False,
        half_truss=False,
        # horizontal_buffer = 0.7,
        # vertical_buffer = 0.2,
        cols=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # mesh reflector
        cap1, rim1 = create_scalloped_cap(
            radius=radius_mesh_base,
            height=height,
            num_scallops=num_scallops,
            scallop_depth=scallop_depth,
        )
        cap1.set_fill(GOLD_D, opacity=1)
        cap1.set_stroke(width=0)
        cap1.z_index = mesh_z_index
        # The mesh is the top of a sphere with radius 15, so we have to shift
        # it down (it is currently not at the origin).
        shift_mesh = radius_mesh_base - 1
        cap1.shift(IN * shift_mesh)
        self.cap = cap1

        cylinders = create_truss_opengl(
            num_scallops=num_scallops,
            rim1=rim1,
            shift_mesh=shift_mesh,
            height=height,
            full_range=full_truss,
            half_truss=half_truss,
        )

        solar_array = create_solar_array_opengl()
        solar_array.shift(2.5 * OUT + 6 * UP)
        if not back_view:
            self.add(cap1)
        self.add(*cylinders)
        self.add(*solar_array)

        # Box spans across both middle rectangles
        box_width = 0.75  # ((col2_x - col1_x) + rect_width)/2
        box_height = 1.3  # Adjust as needed
        box_depth = 0.5
        box = Prism(dimensions=[1, 1, 1]).set_fill(GREY_E, opacity=1)
        box.stretch(box_width, dim=0)  # x
        box.stretch(box_depth, dim=1)  # y
        box.stretch(box_height, dim=2)  # z

        # Position box at bottom of solar array
        box.next_to(solar_array, IN, buff=0.1)
        box.z_index = box_z_index
        for cyl in [*cylinders]:
            cyl.z_index = cyl_z_index
        for rect in [*solar_array]:
            rect.z_index = solar_array_z_index

        # Calculate cylinder dimensions
        cylinder_radius = 0.07
        cylinder_height = 3.5  # How far along IN direction it extends

        # Create cylinder (default orientation is along Z-axis, which is IN/OUT)
        cylinder = Cylinder(radius=cylinder_radius, height=cylinder_height)

        # Position at bottom of box
        cylinder.next_to(box, IN, buff=0)
        cylinder.set_color(GREY_D)
        self.add(cylinder)

        # Calculate second cylinder dimensions
        cylinder2_radius = 0.07  # Same as first cylinder
        cylinder2_height = 1  # Length of angled cylinder

        # Create second cylinder
        cylinder2 = Cylinder(radius=cylinder2_radius, height=cylinder2_height)

        # Calculate second cylinder dimensions
        cylinder2_radius = 0.07
        cylinder2_height = 2

        # Create second cylinder
        cylinder2 = Cylinder(radius=cylinder2_radius, height=cylinder2_height)

        # Define the angle
        angle = -PI / 2.75

        # Rotate cylinder2 first (before positioning)
        cylinder2.rotate(angle, axis=RIGHT)

        # Calculate the direction vector of the rotated cylinder
        # Cylinder extends from -height/2 to +height/2 along this direction
        direction = np.array([0, np.sin(angle), -np.cos(angle)])

        # Position cylinder at connection point
        connection_point = cylinder.get_center() + (cylinder_height / 2) * IN
        cylinder2.move_to(connection_point + (cylinder2_height / 2) * direction)
        cylinder2.set_color(GREY_D)

        self.add(cylinder2)
        self.add(box)  # renders box on top of cylinder

        # Create box at bottom of second cylinder
        box2_width = 0.5
        box2_height = 0.3
        box2_depth = 0.3

        box2 = Prism(dimensions=[1, 1, 1])
        box2.stretch(box2_width, dim=0)  # x
        box2.stretch(box2_depth, dim=1)  # y
        box2.stretch(box2_height, dim=2)  # z

        # Position box at bottom of cylinder 2
        box2_position = cylinder2.get_center() + (cylinder2_height / 2) * direction
        box2.move_to(box2_position)
        box2.set_color(GREY_E)
        self.add(box2)
        # render mesh on top for opengl front view
        if back_view:
            self.add(cap1)

    def create_scalloped_cap(
        self,
        radius=20.0,
        height=2.0,
        num_scallops=6,
        scallop_depth=1,
        mesh_res=[40, 40],
    ):
        """
        Spherical cap with scalloped cutouts carved from the bottom,
        while preserving a perfect spherical surface.
        """

        # Fixed cap angle
        base_z_min = radius - height
        phi_cap = np.arccos(base_z_min / radius)

        def scalloped_phi_cut(v):
            angle_step = TAU / num_scallops
            t = ((v + angle_step / 2) % angle_step) / angle_step

            # triangular wave → sharp peaks
            sharp = abs(np.sin(t * TAU))

            z_cut = base_z_min + scallop_depth * sharp
            return np.arccos(z_cut / radius)

        def rim_point(v):
            # 3D point at u=1 for a given angle v
            phi = scalloped_phi_cut(v)
            return radius * np.array(
                [np.sin(phi) * np.cos(v), np.sin(phi) * np.sin(v), np.cos(phi)]
            )

        cap = Surface(
            lambda u, v: (
                radius
                * np.array(
                    [
                        np.sin(min(u * phi_cap, scalloped_phi_cut(v))) * np.cos(v),
                        np.sin(min(u * phi_cap, scalloped_phi_cut(v))) * np.sin(v),
                        np.cos(min(u * phi_cap, scalloped_phi_cut(v))),
                    ]
                )
            ),
            u_range=[0, 1],
            v_range=[0, TAU],
            resolution=(
                mesh_res[0],
                num_scallops * 10,
            ),
        )

        return cap, rim_point


def MeshReflectorSatSideView(
    radius=2, num_scallops=12, scallop_depth=0.9, rotation_offset=True
):
    """Satellite with a mesh reflector antenna and solar panels (side view)."""
    # Calculate the radius of the scallop circles
    angle_step = TAU / num_scallops

    # Rotate by half a step to put a low/indent at the top instead of a tip
    if rotation_offset:
        rotation_offset = angle_step / 2 + 0.15

    # The scallop circles need to be positioned OUTSIDE so arcs bend inward
    scallop_center_distance = radius + scallop_depth

    arcs = VGroup()

    for i in range(num_scallops):
        angle = i * angle_step + rotation_offset
        next_angle = (i + 1) * angle_step + rotation_offset

        # Position of this scallop's center (pushed outward)
        center_x = scallop_center_distance * np.cos(angle + angle_step / 2)
        center_y = scallop_center_distance * np.sin(angle + angle_step / 2)

        # Calculate start and end points on the outer circle
        start_point = radius * np.array([np.cos(angle), np.sin(angle), 0])
        end_point = radius * np.array([np.cos(next_angle), np.sin(next_angle), 0])

        # Create arc from start to end point
        scallop_center = np.array([center_x, center_y, 0])

        to_start = start_point - scallop_center
        to_end = end_point - scallop_center

        start_arc_angle = np.arctan2(to_start[1], to_start[0])
        end_arc_angle = np.arctan2(to_end[1], to_end[0])

        # Calculate arc angle
        arc_angle = end_arc_angle - start_arc_angle
        if arc_angle < 0:
            arc_angle += TAU
        if arc_angle > PI:
            arc_angle -= TAU

        # Calculate scallop radius
        scallop_radius = np.linalg.norm(to_start)

        arc = Arc(
            radius=scallop_radius,
            start_angle=start_arc_angle,
            angle=arc_angle,
            arc_center=scallop_center,
        )

        arcs.add(arc)

    # Combine all arcs into one shape
    scalloped = VMobject()
    scalloped.set_points(arcs[0].points)
    for arc in arcs[1:]:
        scalloped.append_points(arc.points)
    scalloped.close_path()
    # style
    scalloped.set_fill(GOLD_E, opacity=1).set_stroke(BLACK)

    # Create a 2x3 grid of rectangles (2 rows, 3 columns)
    rows = 3
    cols = 2
    rect_width = 0.7
    rect_height = 0.5
    horizontal_buffer = 0.1
    vertical_buffer = 0.07

    rectangles = VGroup()

    for row in range(rows):
        perspective_scale_row = 1 - ((rows - 1 - row) / rows) * 0.1
        for col in range(cols):
            perspective_scale_col = 1 - (col / cols) * 0.1
            rect = Rectangle(width=rect_width, height=rect_height)

            # Position the rectangle in the grid
            x_pos = (
                col * (rect_width + horizontal_buffer)
                - (cols - 1) * (rect_width + horizontal_buffer) / 2
            )
            y_pos = -row * (rect_height + vertical_buffer)

            rect.move_to([x_pos, y_pos, 0]).scale(
                perspective_scale_row * perspective_scale_col
            )
            rectangles.add(rect)

    # Position the grid below the scalloped circle
    rectangles.next_to(scalloped, DOWN, buff=-0.75)

    # Optional: style the rectangles
    rectangles.set_stroke(BLACK, width=2)
    rectangles.set_fill(GREY_D, opacity=1)

    lines = VGroup()
    points = scalloped.get_all_points()

    # Pick every end of each scallop
    step = len(points) // num_scallops

    for idx, i in enumerate(range(0, len(points), step)):
        tip = points[i]
        line = Line(tip, tip + UP * 1.0, color=BLACK)
        if idx not in [6, 7]:
            # the 6th one makes it look a bit messy bc
            # it overlaps with the solar array
            lines.add(line)

    cylinder_width = 0.1  # width in 2D, corresponds to diameter
    cylinder_height = 1.5  # height

    cylinder = Rectangle(width=cylinder_width, height=cylinder_height, fill_opacity=1)
    cylinder.set_fill(GREY_D).set_stroke(BLACK)
    cylinder.rotate(-0.05 * PI)
    center_x = 1.5
    center_y = -2.2
    cylinder.move_to([center_x, center_y + cylinder_height / 2, 0])

    cylinder2_width = 0.1
    cylinder2_height = 0.5

    cylinder2 = Rectangle(
        width=cylinder2_width, height=cylinder2_height, fill_opacity=1
    )
    cylinder2.set_fill(GREY_D)

    # Define angle (radians)
    angle = -PI / 3  # negative = downward/left
    cylinder2.rotate(angle)

    # Connect the top of first cylinder to one end of second
    connection_point = cylinder.get_bottom()  # top of first cylinder
    cylinder2.move_to(connection_point + 0.32 * LEFT + 0.1 * DOWN)

    box_width = 0.45
    box_height = 0.25
    box2 = Rectangle(width=box_width, height=box_height, fill_opacity=1)
    box2.set_fill(GREY_D)

    # Position at the tip of second cylinder
    box2.move_to(
        cylinder2.get_bottom() + 0.4 * LEFT + 0.075 * UP
    )  # 2D top = end along height

    dot = Circle(radius=0.03, fill_opacity=1)
    dot.set_fill(GREY_D).set_stroke(width=0)
    dot.move_to(cylinder2.get_top() + 0.205 * RIGHT + 0.0485 * DOWN).shift(0.15 * UP)

    merged = Union(cylinder, cylinder2, box2, fill_opacity=1)
    merged.set_fill(GREY_D).set_stroke(BLACK).shift(0.15 * UP)

    scalloped.apply_matrix([[-1, 0, 0], [0.1, 0.5, 0], [0, 0, -1]])

    lines.apply_matrix([[-1, 0, 0], [0.1, 0.5, 0], [0, 0, -1]])

    rectangles.apply_matrix([[1, -0.7, 0], [0.1, 1, 0], [0, 0, 1]]).rotate(
        22 * DEGREES
    ).shift(1.8 * UP + -0.1 * RIGHT)

    return VGroup(merged, dot, scalloped, lines, rectangles)


def MeshReflectorSat(radius=2, num_scallops=8, scallop_depth=0.3, rotation_offset=True):
    """Satellite with a mesh reflector antenna and solar panels (top view)."""
    # Calculate the radius of the scallop circles
    angle_step = TAU / num_scallops

    # Rotate by half a step to put a low/indent at the top instead of a tip
    if rotation_offset:
        rotation_offset = angle_step / 2

    # The scallop circles need to be positioned OUTSIDE so arcs bend inward
    scallop_center_distance = radius + scallop_depth

    arcs = VGroup()

    for i in range(num_scallops):
        angle = i * angle_step + rotation_offset
        next_angle = (i + 1) * angle_step + rotation_offset

        # Position of this scallop's center (pushed outward)
        center_x = scallop_center_distance * np.cos(angle + angle_step / 2)
        center_y = scallop_center_distance * np.sin(angle + angle_step / 2)

        # Calculate start and end points on the outer circle
        start_point = radius * np.array([np.cos(angle), np.sin(angle), 0])
        end_point = radius * np.array([np.cos(next_angle), np.sin(next_angle), 0])

        # Create arc from start to end point
        scallop_center = np.array([center_x, center_y, 0])

        to_start = start_point - scallop_center
        to_end = end_point - scallop_center

        start_arc_angle = np.arctan2(to_start[1], to_start[0])
        end_arc_angle = np.arctan2(to_end[1], to_end[0])

        # Calculate arc angle
        arc_angle = end_arc_angle - start_arc_angle
        if arc_angle < 0:
            arc_angle += TAU
        if arc_angle > PI:
            arc_angle -= TAU

        # Calculate scallop radius
        scallop_radius = np.linalg.norm(to_start)

        arc = Arc(
            radius=scallop_radius,
            start_angle=start_arc_angle,
            angle=arc_angle,
            arc_center=scallop_center,
        )

        arcs.add(arc)

    # Combine all arcs into one shape
    scalloped = VMobject()
    scalloped.set_points(arcs[0].points)
    for arc in arcs[1:]:
        scalloped.append_points(arc.points)
    scalloped.close_path()

    # Create a 2x3 grid of rectangles (2 rows, 3 columns)
    rows = 3
    cols = 2
    rect_width = 0.9
    rect_height = 0.6
    horizontal_buffer = 0.7
    vertical_buffer = 0.07

    rectangles = VGroup()

    for row in range(rows):
        for col in range(cols):
            rect = Rectangle(width=rect_width, height=rect_height)

            # Position the rectangle in the grid
            x_pos = (
                col * (rect_width + horizontal_buffer)
                - (cols - 1) * (rect_width + horizontal_buffer) / 2
            )
            y_pos = -row * (rect_height + vertical_buffer)

            rect.move_to([x_pos, y_pos, 0])
            rectangles.add(rect)

    # Position the grid below the scalloped circle
    rectangles.next_to(scalloped, DOWN, buff=-0.75)

    # Optional: style the rectangles
    rectangles.set_stroke(BLACK, width=2)
    rectangles.set_fill(GREY_D, opacity=1)

    # Create rectangle with circle between middle rectangles
    # The middle rectangles are at indices 1 (top middle) and 4 (bottom middle)
    top_middle = rectangles[1]
    bottom_middle = rectangles[4]

    # Create the new rectangle positioned between them
    special_rect = Rectangle(width=0.55, height=0.55)
    special_rect.move_to((top_middle.get_center() + bottom_middle.get_center()) / 2)
    special_rect.set_stroke(BLACK, width=2)
    special_rect.set_fill(GREY, opacity=0.4)

    # Create circle inside the rectangle
    circle = Circle(radius=0.55 * 0.2)
    circle.move_to(special_rect.get_center())
    circle.set_stroke(BLACK, width=2)
    circle.set_fill(BLACK, opacity=0.6)

    scalloped.set_fill(GOLD_E, opacity=1).set_stroke(BLACK)

    return VGroup(scalloped, rectangles, special_rect, circle)
