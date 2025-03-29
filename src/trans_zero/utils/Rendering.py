import pygame
import numpy as np

# A simple transform class to mimic gym's Transform (only supports translation here)
class Transform:
    def __init__(self, translation=(0, 0)):
        self.translation = translation

# A simple geometry wrapper that stores drawing info and an accumulated transform.
class Geom:
    def __init__(self, geom_type, params):
        self.geom_type = geom_type  # "polygon", "circle", or "polyline"
        self.params = params        # dictionary of parameters (points, color, etc.)
        self.transform = (0, 0)     # additional translation to be applied
    def add_attr(self, t):
        # t is expected to be a Transform with a translation attribute (a tuple)
        self.transform = (self.transform[0] + t.translation[0],
                          self.transform[1] + t.translation[1])
        return self

# A minimal viewer that uses pygame to render our objects.
class Viewer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Custom Rendering")
        self.geoms = []  # list of Geom objects to draw
        self.bounds = (0, width, 0, height)  # world coordinate bounds: (xmin, xmax, ymin, ymax)
        self.clock = pygame.time.Clock()

    def set_bounds(self, x_min, x_max, y_min, y_max):
        self.bounds = (x_min, x_max, y_min, y_max)

    def world_to_screen(self, point):
        # Transform a point from world coordinates to screen (pixel) coordinates.
        x, y = point
        xmin, xmax, ymin, ymax = self.bounds
        scale_x = self.width / (xmax - xmin)
        scale_y = self.height / (ymax - ymin)
        screen_x = int((x - xmin) * scale_x)
        # Flip y so that y=0 is at the bottom of the world.
        screen_y = int(self.height - (y - ymin) * scale_y)
        return (screen_x, screen_y)

    def draw_polygon(self, points, color):
        geom = Geom("polygon", {"points": points, "color": color})
        self.geoms.append(geom)
        return geom

    def draw_circle(self, radius, n_segments, color, filled=True, linewidth=1):
        # Here, the center is provided later as a translation attribute.
        geom = Geom("circle", {"radius": radius,
                               "n_segments": n_segments,
                               "color": color,
                               "filled": filled,
                               "linewidth": linewidth})
        self.geoms.append(geom)
        return geom

    def draw_polyline(self, points, color, linewidth=1):
        geom = Geom("polyline", {"points": points,
                                 "color": color,
                                 "linewidth": linewidth})
        self.geoms.append(geom)
        return geom

    def render(self, return_rgb_array=False):
        # Clear the screen with white background.
        self.screen.fill((255, 255, 255))
        # Process each geometry object.
        for geom in self.geoms:
            if geom.geom_type == "polygon":
                # Apply the stored transform to each vertex.
                pts = [(x + geom.transform[0], y + geom.transform[1]) for (x, y) in geom.params["points"]]
                pts = [self.world_to_screen(pt) for pt in pts]
                color = tuple(int(c * 255) for c in geom.params["color"])
                pygame.draw.polygon(self.screen, color, pts)
            elif geom.geom_type == "circle":
                # For circles, we use the transform as the center.
                center = self.world_to_screen(geom.transform)
                # Scale radius using x-scale (assumes x and y scales are similar).
                scale = self.width / (self.bounds[1] - self.bounds[0])
                radius = int(geom.params["radius"] * scale)
                color = tuple(int(c * 255) if isinstance(c, float) else int(c) for c in geom.params["color"])
                if geom.params["filled"]:
                    pygame.draw.circle(self.screen, "red", center, radius)
                else:
                    pygame.draw.circle(self.screen, "red", center, radius, geom.params["linewidth"])
            elif geom.geom_type == "polyline":
                pts = [(x + geom.transform[0], y + geom.transform[1]) for (x, y) in geom.params["points"]]
                pts = [self.world_to_screen(pt) for pt in pts]
                color = tuple(int(c * 255) for c in geom.params["color"])
                pygame.draw.lines(self.screen, color, False, pts, geom.params["linewidth"])
        pygame.display.flip()
        self.clock.tick(60)
        # If an RGB array is requested, grab the current frame.
        if return_rgb_array:
            arr = pygame.surfarray.array3d(self.screen)
            arr = np.transpose(arr, (1, 0, 2))  # convert from (width, height, 3) to (height, width, 3)
        # Clear geoms for the next frame.
        self.geoms = []
        return arr if return_rgb_array else None
