from manim import *
import numpy as np
from PIL import Image
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject, OpenGLVGroup
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
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


class Sat3DScene(ThreeDScene):
    def construct(self):

        # only run this once per resolution:
        # Get camera pixel dimensions
        pw = self.camera.pixel_width
        ph = self.camera.pixel_height

        # Load and resize image to match camera
        img = Image.open("./imgs/background.png")
        img = img.resize((pw, ph), Image.LANCZOS)
        img = img.convert("RGBA")  # Ensure RGBA format

        # Save resized image temporarily
        img.save("./imgs/background_resized.png")

        bg_image = "./imgs/background_resized.png"
        self.camera.background_image = bg_image
        self.camera.init_background()

        # Create the mesh and net
        height = 0.5
        num_scallops = 6
        cap1, rim1 = self.create_scalloped_cap(
            radius=15.0, height=height, num_scallops=num_scallops, scallop_depth=0.1
        )
        cap2, _ = self.create_scalloped_cap(
            radius=15.0, height=height, num_scallops=num_scallops, scallop_depth=0.1
        )

        # style mesh reflector
        cap1.set_fill(GOLD_D, opacity=0.9)
        cap1.set_stroke(GOLD_E, width=1.5)
        # style net
        cap2.set_fill(GOLD_E, opacity=0)
        cap2.set_stroke(GOLD_E, width=1, opacity=0.2)

        # flip net and place it slightly above cap1
        cap2.rotate(PI, axis=UP)
        shift_mesh = 14
        shift_net = 13
        cap1.shift(IN * shift_mesh)
        cap2.shift(IN * shift_net)

        # cylinders that connect the mesh and net
        def cylinder_between(p1, p2, radius=0.1, color=BLUE):
            axis = p2 - p1
            height = np.linalg.norm(axis)
            direction = axis / height

            cyl = Cylinder(
                radius=radius, height=height, direction=direction, color=color
            )

            cyl.move_to((p1 + p2) / 2)
            return cyl

        # Loop over each scallop
        for i in range(num_scallops * 2):
            # angle at start of scallop
            v = i * TAU / num_scallops / 2

            # rim point in local cap coordinates
            p_local1 = rim1(v)

            # convert to world coordinates if cap1 has been shifted
            p_world1 = p_local1
            p_world2 = p_local1

            # create a small dot at that position
            dot1 = Dot(p_world1, radius=0.2, color=RED).shift(IN * shift_mesh)
            dot2 = Dot(p_world2, radius=0.2, color=RED).shift(
                IN * shift_net + (height * 1) * OUT
            )

            cyl = cylinder_between(
                dot1.get_center(), dot2.get_center(), radius=0.03, color=BLACK
            ).set_stroke(BLACK)

            self.add(cyl)  # ,dot1,dot2)

        # Set up camera

        self.add(cap1, cap2)

        # Create a 2x3 grid of rectangles (2 rows, 3 columns)
        rows = 3
        cols = 2
        rect_width = 2
        rect_height = 1
        horizontal_buffer = 0.7
        vertical_buffer = 0.07

        rectangles = VGroup()

        for row in range(rows):
            for col in range(cols):
                # Create a 3D rectangular prism (box)
                rect = Prism(
                    dimensions=[rect_width, rect_height, 0.05]  # width, height, depth
                )

                # Position the rectangle in the grid
                x_pos = (
                    col * (rect_width + horizontal_buffer)
                    - (cols - 1) * (rect_width + horizontal_buffer) / 2
                )
                y_pos = -row * (rect_height + vertical_buffer)

                # Move to position, orthogonal to z-axis (facing OUT)
                rect.move_to([x_pos, y_pos, 0])
                rect.z_index = 1
                rect.set_color(GREY_E).set_fill(
                    opacity=1
                )  # .set_sheen(0.8, direction=UR).set_sheen_color(BLUE_B)
                rectangles.add(rect.shift(2 * OUT + 5 * UP))

        self.add(*rectangles)

        # Create box touching bottom of middle 2 rectangles (columns 1 and 2, row 1)
        # Calculate position of middle 2 rectangles in bottom row
        col1_x = (
            0 * (rect_width + horizontal_buffer)
            - (cols - 1) * (rect_width + horizontal_buffer) / 2
        )
        col2_x = (
            1 * (rect_width + horizontal_buffer)
            - (cols - 1) * (rect_width + horizontal_buffer) / 2
        )
        bottom_row_y = -1 * (rect_height + vertical_buffer)

        # Box spans across both middle rectangles
        box_width = 1.3  # ((col2_x - col1_x) + rect_width)/2
        box_height = 0.75  # Adjust as needed
        box_depth = 1.75
        box = Prism(dimensions=[box_width, box_height, box_depth]).set_fill(
            GREY_E, opacity=1
        )
        box.z_index = 0

        # Position box centered between the two rectangles, touching their bottom
        center_x = (col1_x + col2_x) / 2
        box_y = (
            bottom_row_y - rect_height / 2 - box_height / 2
        )  # Bottom of rectangles minus half box height

        box.move_to([center_x, box_y, 0])
        self.add(box.shift(1.02 * OUT + 6 * UP))

        # Calculate cylinder dimensions
        cylinder_radius = 0.07  # Adjust as needed
        cylinder_height = 3.5  # How far along IN direction it extends

        # Create cylinder (default orientation is along Z-axis, which is IN/OUT)
        cylinder = Cylinder(radius=cylinder_radius, height=cylinder_height)

        # Position at bottom of box
        # Box is at z = 1.05 (from the OUT shift)
        # Cylinder should extend IN from the box, so its center is at:
        cylinder_z = 1.05 - box_depth / 2 - cylinder_height / 2

        cylinder.move_to([center_x, box_y, cylinder_z])
        cylinder.shift(6 * UP)  # Only apply the UP shift, Z position already set

        # Optional: set color
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

        # Find connection point (end of first cylinder)
        connection_point = cylinder.get_center() + (cylinder_height / 2) * IN

        # Position cylinder2 so its top end touches the connection point
        # Top end is at center - (cylinder2_height/2) * direction
        # So: connection_point = cylinder2.center - (cylinder2_height/2) * direction
        # Therefore: cylinder2.center = connection_point + (cylinder2_height/2) * direction
        cylinder2.move_to(connection_point + (cylinder2_height / 2) * direction)
        cylinder2.set_color(GREY_D)

        self.add(cylinder2)

        # Create box at bottom of second cylinder
        box2_width = 0.5
        box2_height = 0.3
        box2_depth = 0.3

        box2 = Prism(dimensions=[box2_width, box2_height, box2_depth])

        # Find the bottom of cylinder2
        # Cylinder2 center + half its length along its direction
        box2_position = cylinder2.get_center() + (cylinder2_height / 2) * direction

        box2.move_to(box2_position)

        # Optional: set color
        box2.set_color(GREY_E)

        box2.z_index = 2
        self.add(box2)

        # line for (1) mesh reflector, (2) net, (3) solar array,
        # (4) Propulsion, Attitude Control, Data Downlink, command/control
        # (5) high power radar
        # 100-200kg
        # antenna spans approx. 8 sq^2

        # theta 180 degrees = solar panels on left hand side, side view
        # phi = 0 -> top view
        # phi = 90 -> side view
        self.set_camera_orientation(phi=75 * DEGREES, theta=0 * DEGREES, zoom=0.5)
        self.begin_ambient_camera_rotation(
            rate=0.4
        )  # approx 20 seconds for one rotation
        self.wait(1)

        # mesh reflector
        self.stop_ambient_camera_rotation()

        self.move_camera(
            phi=120 * DEGREES, theta=180 * DEGREES, run_time=2, zoom=0.6  # tilt up/down
        )

        # move rectangles back otherwise they overlap with mesh
        for rect in rectangles:
            rect.z_index = 0

        line_2d = VMobject(color=BLACK)
        line_2d.set_points_as_corners(
            [
                [0, 0, 0],  # start
                [1, -0.75, 0],  # diagonal down
                [2, -0.75, 0],  # straight right
            ]
        )

        # Perimeter Truss Deployable Antenna: https://www.mdpi.com/2072-4292/17/14/2432 3.1.2
        text = (
            Text("mesh reflector", font="Zalando Sans", color=BLACK)
            .scale(0.5)
            .set_stroke(BLACK)
        )
        text.next_to(line_2d.get_end(), RIGHT)

        self.add_fixed_in_frame_mobjects(line_2d)
        self.play(Create(line_2d))

        self.add_fixed_in_frame_mobjects(text)
        self.play(FadeIn(text))

        line_2d_radar = VMobject(color=BLACK)
        line_2d_radar.set_points_as_corners(
            [
                [-1.1, -2.45, 0],  # start
                [0.5, -2.45, 0],  # diagonal down
            ]
        )

        # single-beam feed
        # primary beam is shaped by the reflector -> directed beam (can only generate one beam, not multiple
        # beams with different directions)
        text_radar = (
            Text("feed assembly", font="Zalando Sans", color=BLACK)
            .scale(0.5)
            .set_stroke(BLACK)
        )
        text_radar.next_to(line_2d_radar.get_end(), RIGHT)

        self.add_fixed_in_frame_mobjects(line_2d_radar)
        self.play(Create(line_2d_radar))

        self.add_fixed_in_frame_mobjects(text_radar)
        self.play(FadeIn(text_radar))

        self.wait(1)
        self.play(FadeOut(line_2d, text, line_2d_radar, text_radar))

        # camera position net on top
        self.move_camera(phi=60 * DEGREES, theta=180 * DEGREES, zoom=0.6, run_time=2)

        # move rectangles to front again otherwise they overlap with mesh
        for rect in rectangles:
            rect.z_index = 1

        self.wait(1)

        line_2d = VMobject(color=BLACK)
        line_2d.set_points_as_corners(
            [
                [0.25, 1, 0],  # start
                [0.75, 2.5, 0],  # diagonal down
                [2.5, 2.5, 0],  # straight right
            ]
        )
        text = (
            Text("net", font="Zalando Sans", color=BLACK).scale(0.5).set_stroke(BLACK)
        )
        text.next_to(line_2d.get_end(), RIGHT)

        self.add_fixed_in_frame_mobjects(line_2d)
        self.play(Create(line_2d))

        self.add_fixed_in_frame_mobjects(text)
        self.play(FadeIn(text))

        text_solar = (
            Text("solar array", font="Zalando Sans", color=BLACK)
            .scale(0.5)
            .set_stroke(BLACK)
        )
        line_2d_solar = VMobject(color=BLACK)
        line_2d_solar.set_points_as_corners(
            [
                [-3.25, 1.5, 0],  # start
                [-4, 2, 0],  # straight right
                [-4.5, 2, 0],  # end
            ]
        )
        text_solar.next_to(line_2d_solar.get_end(), LEFT)

        self.add_fixed_in_frame_mobjects(line_2d_solar)
        self.play(Create(line_2d_solar))
        self.add_fixed_in_frame_mobjects(text_solar)
        self.play(FadeIn(text_solar))

        self.wait(1)
        self.play(FadeOut(line_2d, text, text_solar, line_2d_solar))

        self.move_camera(
            phi=90 * DEGREES, theta=90 * DEGREES, run_time=2, zoom=0.6  # tilt up/down
        )
        for rect in rectangles:
            rect.z_index = 0
        self.wait(1)

        line_2d = VMobject(color=BLACK)
        line_2d.set_points_as_corners(
            [
                [0, 0.5, 0],  # start
                [1, -1, 0],  # diagonal down
                [2, -1, 0],  # straight right
            ]
        )
        text1 = (
            Text("command/control", font="Zalando Sans", color=BLACK)
            .scale(0.5)
            .set_stroke(BLACK)
        )
        text1.next_to(line_2d.get_end(), RIGHT)

        line_22d = VMobject(color=BLACK)
        line_22d.set_points_as_corners(
            [
                [-0.25, 1, 0],  # start
                [-1, 0, 0],  # diagonal down
                [-2, 0, 0],  # straight right
            ]
        )

        text2 = (
            Text("propulsion, attitude control", font="Zalando Sans", color=BLACK)
            .scale(0.5)
            .set_stroke(BLACK)
        )
        text2.next_to(line_22d.get_end(), LEFT)

        line_32d = VMobject(color=BLACK)
        line_32d.set_points_as_corners(
            [
                [0, 1.25, 0],  # start
                [1, 2.75, 0],  # diagonal down
                [1.75, 2.75, 0],  # straight right
            ]
        )

        text3 = (
            Text("data downlink", font="Zalando Sans", color=BLACK)
            .scale(0.5)
            .set_stroke(BLACK)
        )
        text3.next_to(line_32d.get_end(), RIGHT)

        texts = [text1, text2, text3]
        lines = [line_2d, line_22d, line_32d]

        self.add_fixed_in_frame_mobjects(*lines)
        self.remove(*lines)
        self.play(
            AnimationGroup(
                Create(line_2d), Create(line_22d), Create(line_32d), lag_ratio=0
            )
        )

        self.add_fixed_in_frame_mobjects(*texts)
        self.play(FadeIn(*texts))

        self.wait(2)

        self.play(FadeOut(*texts, *lines))

        self.move_camera(
            phi=75 * DEGREES, theta=180 * DEGREES, run_time=2, zoom=0.6  # tilt up/down
        )

        for rect in rectangles:
            rect.z_index = 1

        self.wait(1)

        self.stop_ambient_camera_rotation()

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


class Satellite(VGroup):
    def __init__(
        self,
        position=ORIGIN,
        color=WHITE,
        stroke_color=PURPLE_E,
        stroke_width=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        sat_body = (
            Cube(
                side_length=2,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                fill_color=color,
            )
            .scale([0.3, 0.3, 0.3])
            # .set_shade_in_3d(True)
            .move_to(position)
            .set_opacity(1)
        )

        # Solar panels (perpendicular to antenna)
        panel1 = (
            Cube(
                side_length=2,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                fill_color=color,
            )
            .scale([0.5, 0.05, 0.2])
            # .set_shade_in_3d(True)
            .next_to(sat_body, LEFT, buff=0.4)
            .set_opacity(1)
        )

        panel2 = (
            Cube(
                side_length=2,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                fill_color=color,
            )
            .scale([0.5, 0.05, 0.2])
            # .set_shade_in_3d(True)
            .next_to(sat_body, RIGHT, buff=0.4)
            .set_opacity(1)
        )

        antenna = (
            Cube(
                side_length=2,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                fill_color=color,
            )
            .scale([1, 0.5, 0.05])
            # .set_shade_in_3d(True)
            .next_to(sat_body.get_center(), 1 * IN, buff=0.3)
            .set_opacity(1)
        )

        self.add(sat_body)
        self.add(antenna)


class TestSat2D(Scene):
    def construct(self):
        self.camera.background_color = LIGHT_BEIGE

        axes = ThreeDAxes()

        sat = MeshReflectorSatSideView().flip(UP)
        self.add(axes, sat)


class TestSat(ThreeDScene):
    def construct(self):
        self.camera.background_color = LIGHT_BEIGE
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        axes = ThreeDAxes()

        sat = Satellite(DOWN + UP)
        self.add(axes, sat)


from manim import *


class RadarAnimation(Scene):
    def construct(self):
        line = Line(
            DOWN * 2 + LEFT * 1, DOWN * 1 + RIGHT * 2, color=RED, stroke_width=3
        )
        self.add(line)

        angle = PI / 2
        origin = ORIGIN

        incoming_dir = normalize(DOWN + LEFT * 0.3)
        incoming_start_angle = np.arctan2(incoming_dir[1], incoming_dir[0]) - angle / 2

        p1, p2 = line.get_start(), line.get_end()
        line_vec = normalize(p2 - p1)
        line_normal = np.array([-line_vec[1], line_vec[0], 0])

        t = np.dot(origin - p1, line_vec)
        foot = p1 + t * line_vec
        hit_dist = np.linalg.norm(foot - origin)

        reflected_dir = (
            incoming_dir - 2 * np.dot(incoming_dir, line_normal) * line_normal
        )
        reflected_start_angle = (
            np.arctan2(reflected_dir[1], reflected_dir[0]) - angle / 2
        )

        # extend p1 and p2 far in the downward normal direction to form the mask
        depth = line_normal * -10  # push far in the "below" direction
        mask = Polygon(
            p1,
            p2,
            p2 + depth,
            p1 + depth,
            fill_color=BLACK,
            fill_opacity=1,
            stroke_width=0,
        )

        # outgoing wave
        tracker = ValueTracker(0)
        wave = Arc(
            radius=0.1,
            angle=angle,
            start_angle=incoming_start_angle,
            color=GREEN,
            stroke_width=5,
            arc_center=origin,
        )
        self.add(wave)

        def wave_updater(m):
            dist = tracker.get_value()
            m.become(
                Arc(
                    radius=0.1 + dist,
                    angle=angle,
                    start_angle=incoming_start_angle,
                    color=GREEN,
                    stroke_width=5,
                    arc_center=origin,
                )
            )

        wave.add_updater(wave_updater)
        self.play(
            tracker.animate.set_value(hit_dist - 0.1), run_time=2, rate_func=linear
        )

        # don't remove wave — add mask on top instead and keep wave growing
        self.add(mask)
        self.add(line)  # keep line visible on top of mask

        def wave_updater_masked(m):
            dist = tracker.get_value()
            m.become(
                Arc(
                    radius=0.1 + dist,
                    angle=angle,
                    start_angle=incoming_start_angle,
                    color=GREEN,
                    stroke_width=5,
                    arc_center=origin,
                )
            )

        wave.remove_updater(wave_updater)
        wave.add_updater(wave_updater_masked)

        # return wave
        return_tracker = ValueTracker(0)
        reflected_origin = origin + 2 * (foot - origin)

        return_wave = Arc(
            radius=0.1,
            angle=angle,
            start_angle=reflected_start_angle,
            color=YELLOW,
            stroke_width=5,
            arc_center=reflected_origin,
        )

        def return_updater(m):
            dist = return_tracker.get_value()
            m.become(
                Arc(
                    radius=0.1 + dist,
                    angle=angle,
                    start_angle=reflected_start_angle,
                    color=YELLOW,
                    stroke_width=5,
                    arc_center=foot,
                ).rotate(17 * DEGREES)
            )

        return_wave.add_updater(return_updater)
        self.add(return_wave)

        self.play(
            tracker.animate.set_value(hit_dist + 2),
            return_tracker.animate.set_value(hit_dist + 4),
            run_time=2,
            rate_func=linear,
        )

        wave.remove_updater(wave_updater_masked)
        return_wave.remove_updater(return_updater)
        self.wait(1)
