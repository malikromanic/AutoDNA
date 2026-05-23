from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#podatkovne strukture za vektorje in trikotnikee
@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Triangle:
    normal: Vec3
    v1: Vec3
    v2: Vec3
    v3: Vec3

    def vertices(self) -> List[Vec3]:
        return [self.v1, self.v2, self.v3]


#nalozi stl datoteko in zazna tip stl
def load_stl(file_path: str) -> List[Triangle]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"STL file not found: {file_path}")

    with open(path, "rb") as file:
        data = file.read()

    if is_binary_stl(data):
        return load_binary_stl(data)

    return load_ascii_stl(data.decode("utf-8", errors="ignore"))


#preveri ali je stl binarni ali ascii
def is_binary_stl(data: bytes) -> bool:

    if len(data) < 84:
        return False

    triangle_count = struct.unpack("<I", data[80:84])[0]
    expected_size = 84 + triangle_count * 50

    if expected_size == len(data):
        return True

    start = data[:5].lower()

    if start == b"solid":
        return False

    return False


#nalozi binarni stl modell
def load_binary_stl(data: bytes) -> List[Triangle]:

    triangles: List[Triangle] = []

    triangle_count = struct.unpack("<I", data[80:84])[0]
    offset = 84

    #prebere vse trikotnike iz datoteke
    for _ in range(triangle_count):

        chunk = data[offset:offset + 50]

        if len(chunk) < 50:
            raise ValueError("Unexpected end of binary STL data")

        values = struct.unpack("<12fH", chunk)

        normal = Vec3(values[0], values[1], values[2])

        v1 = Vec3(values[3], values[4], values[5])
        v2 = Vec3(values[6], values[7], values[8])
        v3 = Vec3(values[9], values[10], values[11])

        triangles.append(Triangle(normal, v1, v2, v3))

        offset += 50

    return triangles


#nalozi ascii stl model
def load_ascii_stl(text: str) -> List[Triangle]:

    triangles: List[Triangle] = []

    #odstrani prazne vrstice
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    i = 0

    #pregleduje vse vrstice ascii stl
    while i < len(lines):

        line = lines[i]

        if line.startswith("facet normal"):

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(f"Invalid facet normal line: {line}")

            normal = Vec3(float(parts[2]), float(parts[3]), float(parts[4]))

            if i + 6 >= len(lines):
                raise ValueError("Incomplete ASCII STL facet")

            if lines[i + 1] != "outer loop":
                raise ValueError(
                    f"Expected 'outer loop', got: {lines[i + 1]}"
                )

            #prebere tri oglisca trikotnika
            v1 = parse_vertex_line(lines[i + 2])
            v2 = parse_vertex_line(lines[i + 3])
            v3 = parse_vertex_line(lines[i + 4])

            if lines[i + 5] != "endloop":
                raise ValueError(
                    f"Expected 'endloop', got: {lines[i + 5]}"
                )

            if lines[i + 6] != "endfacet":
                raise ValueError(
                    f"Expected 'endfacet', got: {lines[i + 6]}"
                )

            triangles.append(Triangle(normal, v1, v2, v3))

            i += 7

        else:
            i += 1

    return triangles


#prebere koordinate enega oglisca
def parse_vertex_line(line: str) -> Vec3:

    parts = line.split()

    if len(parts) != 4 or parts[0] != "vertex":
        raise ValueError(f"Invalid vertex line: {line}")

    return Vec3(float(parts[1]), float(parts[2]), float(parts[3]))


#izracuna aabb modela
def compute_bounding_box(triangles: List[Triangle]) -> Tuple[Vec3, Vec3]:

    if not triangles:
        raise ValueError("Cannot compute bounding box of empty triangle list")

    xs = []
    ys = []
    zs = []

    #shrani vse koordinate modela
    for tri in triangles:
        for vertex in tri.vertices():
            xs.append(vertex.x)
            ys.append(vertex.y)
            zs.append(vertex.z)

    min_point = Vec3(min(xs), min(ys), min(zs))
    max_point = Vec3(max(xs), max(ys), max(zs))

    return min_point, max_point


#izpise stevilo trikotnikov in velikost modela
def print_mesh_info(triangles: List[Triangle]) -> None:

    min_point, max_point = compute_bounding_box(triangles)

    print(f"Number of triangles: {len(triangles)}")

    print(
        f"Bounding box min: "
        f"x={min_point.x}, y={min_point.y}, z={min_point.z}"
    )

    print(
        f"Bounding box max: "
        f"x={max_point.x}, y={max_point.y}, z={max_point.z}"
    )


#zaokrozi koordinate zaradi float napak
def round_vertex(v: Vec3, decimals: int) -> Tuple[float, float, float]:

    return (
        round(v.x, decimals),
        round(v.y, decimals),
        round(v.z, decimals),
    )


#ustvari neusmerjen rob
def make_undirected_edge(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:

    return tuple(sorted((a, b)))


#vrne vse robove trikotnika
def triangle_edges_rounded(
    tri: Triangle,
    decimals: int,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:

    rv1 = round_vertex(tri.v1, decimals)
    rv2 = round_vertex(tri.v2, decimals)
    rv3 = round_vertex(tri.v3, decimals)

    return [
        make_undirected_edge(rv1, rv2),
        make_undirected_edge(rv2, rv3),
        make_undirected_edge(rv3, rv1),
    ]


#preveri vodotesnost modela
def check_watertightness(triangles: List[Triangle], decimals: int = 6) -> None:

    edge_counter: Counter = Counter()

    #presteje vse robove v modelu
    for tri in triangles:
        for edge in triangle_edges_rounded(tri, decimals):
            edge_counter[edge] += 1

    total_unique_edges = len(edge_counter)

    #rob mora pripadati natancno dvema trikotnikoma
    good_edges = {
        edge: count for edge, count in edge_counter.items() if count == 2
    }

    bad_edges = {
        edge: count for edge, count in edge_counter.items() if count != 2
    }

    print("\nWatertightness check")
    print("--------------------")
    print(f"Rounding decimals: {decimals}")
    print(f"Total unique edges: {total_unique_edges}")

    print(
        "Number of good edges "
        f"(used exactly 2 times): {len(good_edges)}"
    )

    print(f"Number of bad edges (used != 2 times): {len(bad_edges)}")
    print(f"Mesh is watertight: {len(bad_edges) == 0}")

    #izpise napacne robove
    if bad_edges:

        print("\nFirst 20 bad edges:")

        for index, (edge, count) in enumerate(bad_edges.items(), start=1):

            if index > 20:
                break

            print(f"{index}. edge={edge}, used_by_triangles={count}")


#vizualizacija trikotniskega modela
def visualize_stl(triangles: List[Triangle]) -> None:

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")

    faces = []

    #pripravi vse trikotnike za prikaz
    for tri in triangles:
        faces.append([tri.v1.as_tuple(), tri.v2.as_tuple(), tri.v3.as_tuple()])

    #ustvari 3d mrezo trikotnikov
    mesh = Poly3DCollection(
        faces,
        facecolor="cornflowerblue",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.55,
    )

    axis.add_collection3d(mesh)

    min_point, max_point = compute_bounding_box(triangles)

    #nastavi pravilno razmerje osi
    set_axes_equal_from_bounds(axis, min_point, max_point)

    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title("STL Viewer")

    plt.tight_layout()
    plt.show()


#nastavi enako merilo vseh osi
def set_axes_equal_from_bounds(ax, min_pt: Vec3, max_pt: Vec3) -> None:

    x_mid = (min_pt.x + max_pt.x) / 2.0
    y_mid = (min_pt.y + max_pt.y) / 2.0
    z_mid = (min_pt.z + max_pt.z) / 2.0

    x_size = max_pt.x - min_pt.x
    y_size = max_pt.y - min_pt.y
    z_size = max_pt.z - min_pt.z

    max_range = max(x_size, y_size, z_size) / 2.0

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)


#glavni program
def main() -> None:

    file_path = "model.stl"
    rounding_decimals = 6

    #nalozi model
    triangles = load_stl(file_path)

    #izpise informacije o modelu
    print_mesh_info(triangles)

    #preveri vodotesnost
    check_watertightness(triangles, decimals=rounding_decimals)

    #prikaze model
    visualize_stl(triangles)


if __name__ == "__main__":
    main()