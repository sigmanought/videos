import os

import numpy as np
from manim import *
from manim.mobject.opengl.opengl_geometry import OpenGLPolygon
from manim.mobject.opengl.opengl_image_mobject import OpenGLImageMobject
from manim.mobject.opengl.opengl_surface import OpenGLSurface, OpenGLTexturedSurface
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from PIL import Image
from utils.sat import OpenGLMeshReflectorSat3d

SOURCE_COLOR = ManimColor("#746458")
SOURCE_SCALE = 0.3
SOURCE_FONT_SIZE = 50
RES_TYPE = "low"
WASHED_GREY = ManimColor("#91847c")

# Replace this with the actual SAR image
FILES_DIR = "./02_bridges_in_stripmap/pngs/"
SAR_IMAGE_PATH = f"{FILES_DIR}/placeholder.png"


def crop_image_to_range(
    image_path: str,
    full_u_range: list,
    full_v_range: list,
    current_u_range: list,
    current_v_range: list,
    output_path: str,
):
    """Crop image to match the currently revealed UV range."""
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    full_u_min, full_u_max = full_u_range
    full_v_min, full_v_max = full_v_range

    # Normalize current range to pixel coordinates
    u_frac_min = (current_u_range[0] - full_u_min) / (full_u_max - full_u_min)
    u_frac_max = (current_u_range[1] - full_u_min) / (full_u_max - full_u_min)
    v_frac_min = (current_v_range[0] - full_v_min) / (full_v_max - full_v_min)
    v_frac_max = (current_v_range[1] - full_v_min) / (full_v_max - full_v_min)

    left = int(u_frac_min * w)
    right = int(u_frac_max * w)
    # PIL y-axis is flipped
    top = int((1 - v_frac_max) * h)
    bottom = int((1 - v_frac_min) * h)

    cropped = img.crop((left, top, right, bottom))
    cropped.save(output_path)
    return output_path


class EllipticalBeam(Group):
    def __init__(
        self,
        v_range,
        width=0.2,
        height=0.5,
        footprint_center=(0, 0, 0),
        pos=(0, 0, 0),
        color=WHITE,
        resolution=(32, 32),
        **kwargs,
    ):
        super().__init__(**kwargs)
        pos = np.array(pos, dtype=float)
        footprint_center = np.array(footprint_center, dtype=float)

        # Direction from satellite to footprint center
        direction = footprint_center - pos
        direction_norm = direction / np.linalg.norm(direction)

        # Build a local coordinate frame perpendicular to the beam direction
        # Pick an arbitrary up vector that's not parallel to direction
        up = (
            np.array([0, 1, 0]) if abs(direction_norm[1]) < 0.9 else np.array([1, 0, 0])
        )
        right = np.cross(direction_norm, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, direction_norm)
        up /= np.linalg.norm(up)

        def beam_point(u, v):
            # Center of the ellipse at this v: interpolate from pos to footprint_center
            center = pos + v * direction
            # Ellipse in the plane perpendicular to the beam
            offset = (width / 2) * np.cos(u) * right + (height / 2) * np.sin(u) * up
            return center + v * offset  # v scales the ellipse so tip is a point

        self.beam_surface = OpenGLSurface(
            beam_point,
            u_range=[0, 2 * PI],
            v_range=[v_range[0], v_range[1]],
            resolution=resolution,
            gloss=0,
            sheen=0,
            color=WHITE,
            opacity=1,
        )
        self.add(self.beam_surface)


# this is not an opengl scene, render it without the opengl flag.
class Scene2_p1(ThreeDScene):
    def construct(self):
        self.camera.background_color = "#e6d8bc"
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
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES, zoom=0.5)
        self.begin_ambient_camera_rotation(
            rate=0.4
        )  # approx 20 seconds for one rotation
        self.wait(7)

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
        self.play(Create(line_2d), run_time=1)

        self.add_fixed_in_frame_mobjects(text)
        self.play(FadeIn(text), run_time=1)

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
        self.play(Create(line_2d_radar), run_time=1)

        self.add_fixed_in_frame_mobjects(text_radar)
        self.play(FadeIn(text_radar), run_time=1)

        self.wait(8)
        self.play(FadeOut(line_2d, text, line_2d_radar, text_radar), run_time=1)
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
        self.play(Create(line_2d), run_time=1)
        self.add_fixed_in_frame_mobjects(text)
        self.play(FadeIn(text), run_time=1)

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
        self.play(FadeOut(line_2d, text, text_solar, line_2d_solar), run_time=1)
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
            ),
            run_time=1,
        )

        self.add_fixed_in_frame_mobjects(*texts)
        self.play(FadeIn(*texts), run_time=1)

        self.wait(4)
        self.play(FadeOut(*texts, *lines), run_time=1)

        self.move_camera(
            phi=75 * DEGREES, theta=180 * DEGREES, run_time=2, zoom=0.6  # tilt up/down
        )

        for rect in rectangles:
            rect.z_index = 1

        self.wait(1)

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
        while preserving a spherical surface.
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


# opengl scene, render with flags --renderer=opengl --format=mp4 --disable_caching
class Scene2_p2(ThreeDScene):
    def construct(self):
        self.renderer.background_color = "#e6d8bc"

        label = Text(
            "Stripmap",
            font="Zalando Sans SemiExpanded",
            font_size=SOURCE_FONT_SIZE,
            color=BLACK,
        )
        label.to_corner(DR)  # DR = Down Right

        # sar image
        u_min, u_max_target = 1.15, -4.3
        u_max_tracker = ValueTracker(u_min)
        full_img = np.array(Image.open(SAR_IMAGE_PATH).convert("RGB"))

        # In opengl it's not straightforward to fade in an
        # image object from left to right. Instead we'll
        # crop the image several times and only show the crop
        # This appears as if the image is fading in from left to right.
        # Path to temp file that will be overwritten every frame
        output_dir = "./sar_bridges/scenes/files/progressive_slices"
        os.makedirs(output_dir, exist_ok=True)

        last_width = {"value": -1}  # cache last width to avoid rewriting same slice

        def make_surface():
            """Fade in surface with a crop of an image image on top. This
            appears as if the image is fading in from left to right."""
            alpha = (u_max_tracker.get_value() - u_min) / (u_max_target - u_min)
            alpha = np.clip(alpha, 0, 1)

            W = full_img.shape[1]
            new_W = max(1, int(alpha * W))

            tmp_img_path = os.path.join(output_dir, f"current_slice.png")
            subset = full_img[:, -new_W:, :]  # slice from left
            Image.fromarray(subset).save(tmp_img_path)
            last_width["value"] = new_W

            surf = OpenGLSurface(
                lambda u, v: [u, v, 0],
                u_range=[u_max_tracker.get_value(), u_min],
                v_range=[-1.2, 1.2],
                color=WHITE,
            )

            tex = OpenGLTexturedSurface(
                uv_surface=surf,
                image_file=tmp_img_path,
            ).shift(1 * DOWN + 3 * IN)

            return tex

        # background map (and its oultine) of San Francisco area
        self.set_camera_orientation(phi=70 * DEGREES, theta=160 * DEGREES)
        self.camera.scale(1 / 0.7)
        svg_fill = SVGMobject(f"{FILES_DIR}/background_san_francisco_rotated.svg")
        svg_outline = SVGMobject(f"{FILES_DIR}/background_san_francisco_outline.svg")

        shift_x = 0.9
        for svg in [svg_fill, svg_outline]:
            svg.set_height(6).set_height(3).scale(2)
            svg.move_to(ORIGIN)
            svg.rotate(180 * DEGREES)
            svg.shift(5 * UP).rotate(-12 * DEGREES, axis=OUT).shift(shift_x * RIGHT)

        svg_fill.set(height=3)
        svg_outline.scale(svg_fill.width / svg_outline.width)
        svg_outline.move_to(svg_fill)
        svg_fill.set_fill(WASHED_GREY, opacity=1)
        svg_outline.set_fill(BLACK, opacity=0).set_stroke(BLACK, width=5, opacity=0)
        self.add(svg_fill, svg_outline)

        duration = 0.5  # seconds
        fps = 60
        steps = int(duration * fps)

        for i in range(steps + 1):
            alpha = i / steps
            svg_fill.set_fill(WASHED_GREY, opacity=alpha)
            svg_outline.set_stroke(BLACK, width=5, opacity=alpha)
            self.wait(1 / fps)

        sat = OpenGLMeshReflectorSat3d(back_view=True, half_truss=True).scale(0.4)
        sat.rotate(15 * DEGREES, axis=LEFT)
        sat.rotate(0.5 * DEGREES, axis=DOWN)  # tilt left right
        sat.rotate(190 * DEGREES, axis=OUT)  # rotate left right
        y_index_sat = -4 - 1 - 2
        z_index_sat = 1.95 - 0.7
        sat.move_to([7, y_index_sat, z_index_sat])  # order: xyz

        v_min, v_max = -2.15, 0.15
        z_surface = -3
        dv = 0.8

        def make_trapezoid_3d(sat_pos):
            u_sat, v_sat, z_sat = sat_pos

            vertices = [
                # left hand side
                [u_sat + dv / 2, v_sat - 2, z_sat],
                [u_sat + dv / 2, v_sat - 2, z_sat],
                [u_sat + 0.5, v_max + 1, z_surface],  # front left
                [u_sat + 0.5, v_min + 1, z_surface],  # back left
                # front side
                [u_sat + dv / 2, v_sat - 2, z_sat],
                [u_sat, v_sat - 2, z_sat],
                [u_sat - 0.5, v_max + 1, z_surface],  # front left
                [u_sat + 0.5, v_max + 1, z_surface],  # front right
            ]

            return OpenGLPolygon(*vertices, color=WHITE, fill_opacity=0.7).set_stroke(
                width=0
            )

        sat_pos_orig = np.array(
            [7 + shift_x, -3, 1.95]
        )  # footprint matches image coords.
        trapezoid = make_trapezoid_3d(sat_pos_orig)  # sat.get_center())

        # 7 -> 1.25, length of 6.75
        path = Line(
            start=[7, y_index_sat, z_index_sat], end=[u_min, y_index_sat, z_index_sat]
        )

        # fade in sat
        duration = 0.5
        fps = 60
        steps = int(duration * fps)
        sat.set_opacity(0)
        self.add(sat)
        for i in range(steps + 1):
            alpha = i / steps
            sat.set_opacity(alpha)
            self.wait(1 / fps)

        self.play(MoveAlongPath(sat, path), run_time=5.91, rate_func=linear)

        num_steps = 20
        u_values = np.linspace(u_min, u_max_target, num_steps)

        # Precompute satellite and trapezoid positions along their paths
        sat_start = np.array([u_min, y_index_sat, z_index_sat])
        sat_end = np.array([u_max_target, y_index_sat, z_index_sat])

        # trapezoid path
        trap_start = np.array([u_min, -1.95, 0])
        trap_end = np.array([u_max_target, -1.95, 0])

        previous_surfaces = []

        self.add(trapezoid)

        # length of 3.5, imaging time 4sec
        # draw new image every 4/num_steps sec
        # a bit blocky but relatively short runtime
        for i, u_max in enumerate(u_values):
            # Update surface
            u_max_tracker.set_value(u_max)
            tex = make_surface()
            self.add(tex)
            self.add(trapezoid)
            self.add(sat)
            previous_surfaces.append(tex)

            # Remove previous surface to avoid too many objects
            if len(previous_surfaces) > 1:
                self.remove(previous_surfaces[-2])

            # Interpolation parameter
            t0 = i / num_steps
            t1 = (i + 1) / num_steps

            sat_pos_0 = sat_start * (1 - t0) + sat_end * t0
            sat_pos_1 = sat_start * (1 - t1) + sat_end * t1

            trap_pos_0 = trap_start * (1 - t0) + trap_end * t0
            trap_pos_1 = trap_start * (1 - t1) + trap_end * t1

            # Build short path segments
            sat_path = Line(sat_pos_0, sat_pos_1)
            trap_path = Line(trap_pos_0, trap_pos_1)

            # Animate along paths
            if i == 0:
                img_source = (
                    Text(
                        "Image source: Capella Space, CC BY 4.0 (creativecommons.org/licenses/by/4.0/)",
                        font_size=SOURCE_FONT_SIZE,
                        font="Zalando Sans",
                    )
                    .set_color(SOURCE_COLOR)
                    .scale(SOURCE_SCALE)
                )
                img_source.to_corner(DL).shift(0.15 * DOWN)
                img_source.fix_in_frame()

                self.play(
                    FadeIn(img_source),
                    MoveAlongPath(sat, sat_path),
                    MoveAlongPath(trapezoid, trap_path),
                    run_time=4 / num_steps,
                    rate_func=linear,
                )

            else:
                self.play(
                    MoveAlongPath(sat, sat_path),
                    MoveAlongPath(trapezoid, trap_path),
                    run_time=4 / num_steps,
                    rate_func=linear,
                )

        self.remove(trapezoid)
        path = Line(
            start=[u_max_target, y_index_sat, z_index_sat],
            end=[-7, y_index_sat, z_index_sat],
        )
        distance = np.linalg.norm(path.get_end() - path.get_start())
        speed = distance / 2.843
        self.play(MoveAlongPath(sat, path), run_time=2.843, rate_func=linear)

        # fade in label
        t1 = 1  # same as wait time or animation duration
        d1 = speed * t1
        new_path1 = Line(
            start=sat.get_center(),
            end=sat.get_center() + np.array([-d1, 0, 0]),  # same direction as before
        )
        self.add_fixed_in_frame_mobjects(label)
        self.play(
            MoveAlongPath(sat, new_path1), FadeIn(label), run_time=t1, rate_func=linear
        )

        # wait time
        t2 = 2
        d2 = speed * t2
        new_path2 = Line(
            start=sat.get_center(), end=sat.get_center() + np.array([-d2, 0, 0])
        )

        self.play(MoveAlongPath(sat, new_path2), run_time=t2, rate_func=linear)

        t3 = 1  # fadeout
        d3 = speed * t3
        new_path3 = Line(
            start=sat.get_center(), end=sat.get_center() + np.array([-d3, 0, 0])
        )

        self.play(MoveAlongPath(sat, new_path3), run_time=t3, rate_func=linear)


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
