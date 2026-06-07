#uvoz knjiznic za delo z stl modeli,
#binarnimi podatki in 3d prikazom
from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#razred za predstavitev 3d tocke
@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    #pretvorba tocke v tuple obliko
    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


#razred za predstavitev stl trikotnika
@dataclass
class Triangle:
    normal: Vec3
    v1: Vec3
    v2: Vec3
    v3: Vec3

    #vrne vsa tri oglisca trikotnika
    def vertices(self) -> List[Vec3]:
        return [self.v1, self.v2, self.v3]


#nalaganje stl datoteke in zaznava formata
def load_stl(file_path: str) -> List[Triangle]:
    path = Path(file_path)

    #preveri ce datoteka obstaja
    if not path.exists():
        raise FileNotFoundError(f"STL file not found: {file_path}")

    #branje vsebine datoteke
    with open(path, "rb") as file:
        data = file.read()

    #ce je binary uporabi binary parser
    if is_binary_stl(data):
        return load_binary_stl(data)

    #drugace uporabi ascii parser
    return load_ascii_stl(data.decode("utf-8", errors="ignore"))


#preverjanje ali je datoteka binary stl
def is_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False

    #prebere stevilo trikotnikov iz glave
    triangle_count = struct.unpack("<I", data[80:84])[0]

    #izracun pricakovane velikosti datoteke
    expected_size = 84 + triangle_count * 50

    if expected_size == len(data):
        return True

    #ascii stl se pogosto zacne z solid
    if data[:5].lower() == b"solid":
        return False

    return False


#branje binary stl datoteke
def load_binary_stl(data: bytes) -> List[Triangle]:
    triangles: List[Triangle] = []

    triangle_count = struct.unpack("<I", data[80:84])[0]
    offset = 84

    #prehajanje cez vse trikotnike
    for _ in range(triangle_count):
        chunk = data[offset:offset + 50]

        if len(chunk) < 50:
            raise ValueError("Unexpected end of binary STL data")

        #razpakiranje binarnih podatkov
        values = struct.unpack("<12fH", chunk)

        #normala in oglisca trikotnika
        normal = Vec3(values[0], values[1], values[2])
        v1 = Vec3(values[3], values[4], values[5])
        v2 = Vec3(values[6], values[7], values[8])
        v3 = Vec3(values[9], values[10], values[11])

        triangles.append(Triangle(normal, v1, v2, v3))
        offset += 50

    return triangles


#branje ascii stl datoteke
def load_ascii_stl(text: str) -> List[Triangle]:
    triangles: List[Triangle] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    i = 0

    #prehajanje cez vse vrstice
    while i < len(lines):
        line = lines[i]

        #zacetek facet bloka
        if line.startswith("facet normal"):
            parts = line.split()

            if len(parts) != 5:
                raise ValueError(f"Invalid facet normal line: {line}")

            #branje normale
            normal = Vec3(float(parts[2]), float(parts[3]), float(parts[4]))

            if i + 6 >= len(lines):
                raise ValueError("Incomplete ASCII STL facet")

            if lines[i + 1] != "outer loop":
                raise ValueError(f"Expected 'outer loop', got: {lines[i + 1]}")

            #branje vseh treh vertexov
            v1 = parse_vertex_line(lines[i + 2])
            v2 = parse_vertex_line(lines[i + 3])
            v3 = parse_vertex_line(lines[i + 4])

            if lines[i + 5] != "endloop":
                raise ValueError(f"Expected 'endloop', got: {lines[i + 5]}")

            if lines[i + 6] != "endfacet":
                raise ValueError(f"Expected 'endfacet', got: {lines[i + 6]}")

            triangles.append(Triangle(normal, v1, v2, v3))
            i += 7
        else:
            i += 1

    return triangles


#prebere vertex vrstico oblike vertex x y z
def parse_vertex_line(line: str) -> Vec3:
    parts = line.split()

    if len(parts) != 4 or parts[0] != "vertex":
        raise ValueError(f"Invalid vertex line: {line}")

    return Vec3(float(parts[1]), float(parts[2]), float(parts[3]))


#izracun minimalnih in maksimalnih koordinat modela
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

    return Vec3(min(xs), min(ys), min(zs)), Vec3(max(xs), max(ys), max(zs))


#izpis osnovnih informacij o mrezi
def print_mesh_info(triangles: List[Triangle]) -> None:
    min_point, max_point = compute_bounding_box(triangles)

    print(f"Number of triangles: {len(triangles)}")
    print(f"Bounding box min: x={min_point.x}, y={min_point.y}, z={min_point.z}")
    print(f"Bounding box max: x={max_point.x}, y={max_point.y}, z={max_point.z}")


#funkcije za obdelavo robov modela
def round_vertex(v: Vec3, decimals: int) -> Tuple[float, float, float]:
    return (round(v.x, decimals), round(v.y, decimals), round(v.z, decimals))


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


#preverjanje vodotesnosti mreze
#vsak rob mora biti uporabljen dvakrat
def check_watertightness(triangles: List[Triangle], decimals: int = 6) -> None:
    edge_counter: Counter = Counter()

    #steje uporabo vseh robov
    for tri in triangles:
        for edge in triangle_edges_rounded(tri, decimals):
            edge_counter[edge] += 1

    #isci robove ki niso uporabljeni dvakrat
    bad_edges = {
        edge: count for edge, count in edge_counter.items() if count != 2
    }

    print("\nWatertightness check")
    print("--------------------")
    print(f"Rounding decimals: {decimals}")
    print(f"Total unique edges: {len(edge_counter)}")
    print(f"Number of bad edges: {len(bad_edges)}")
    print(f"Mesh is watertight: {len(bad_edges) == 0}")

    #izpis problematicnih robov
    if bad_edges:
        print("\nFirst 20 bad edges:")
        for index, (edge, count) in enumerate(bad_edges.items(), start=1):
            if index > 20:
                break
            print(f"{index}. edge={edge}, used_by_triangles={count}")


#3d prikaz stl modela
def visualize_stl(triangles: List[Triangle]) -> None:
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")

    faces = []

    #ustvari seznam vseh ploskev
    for tri in triangles:
        faces.append([tri.v1.as_tuple(), tri.v2.as_tuple(), tri.v3.as_tuple()])

    #ustvari 3d mrezo modela
    mesh = Poly3DCollection(
        faces,
        facecolor="cornflowerblue",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.55,
    )

    axis.add_collection3d(mesh)

    #nastavi pravilno razmerje osi
    min_point, max_point = compute_bounding_box(triangles)
    set_axes_equal_from_bounds(axis, min_point, max_point)

    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title("STL model")

    plt.tight_layout()
    plt.show()


#nastavitev enakega razmerja osi
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


#glavna funkcija programa
def main() -> None:
    file_path = "model.stl"
    rounding_decimals = 6

    #nalaganje modela
    triangles = load_stl(file_path)

    #izpis informacij modela
    print_mesh_info(triangles)

    #preverjanje vodotesnosti
    check_watertightness(triangles, decimals=rounding_decimals)

    #3d prikaz modela
    visualize_stl(triangles)


#zacetek izvajanja programa
if __name__ == "__main__":
    main()