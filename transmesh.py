import os
from PIL import Image

def load_obj_with_uvs(obj_path):
    vertices = []
    texcoords = []
    faces = []

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append(tuple(map(float, parts[1:4])))
            elif line.startswith('vt '):
                parts = line.strip().split()
                texcoords.append(tuple(map(float, parts[1:3])))
            elif line.startswith('f '):
                face = []
                for vert in line.strip().split()[1:]:
                    vals = vert.split('/')
                    v_idx = int(vals[0]) - 1
                    vt_idx = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else None
                    face.append((v_idx, vt_idx))
                faces.append(face)

    return vertices, texcoords, faces

def sample_texture(texture_path, texcoords):
    img = Image.open(texture_path).convert('RGB')
    width, height = img.size
    colors = []

    for (u, v) in texcoords:
        px = int(u * width)
        py = int((1 - v) * height)
        r, g, b = img.getpixel((min(px, width - 1), min(py, height - 1)))
        colors.append((r / 255.0, g / 255.0, b / 255.0))

    return colors

def write_colored_obj(output_path, vertices, faces, vertex_colors):
    with open(output_path, 'w') as f:
        for i, (x, y, z) in enumerate(vertices):
            r, g, b = vertex_colors[i]
            f.write(f"v {x} {y} {z} {r:.6f} {g:.6f} {b:.6f}\n")
        for face in faces:
            f.write("f " + " ".join([f"{vi+1}" for vi, _ in face]) + "\n")

def convert_to_colored_obj(obj_path, texture_path, output_path):
    vertices, texcoords, faces = load_obj_with_uvs(obj_path)

    # 为每个顶点分配一个 UV 坐标（只取第一个遇到的）
    vert_to_uv = [None] * len(vertices)
    for face in faces:
        for vi, vti in face:
            if vti is not None and vert_to_uv[vi] is None:
                vert_to_uv[vi] = texcoords[vti]

    # 没有 UV 时默认取中心色
    default_uv = (0.5, 0.5)
    uv_for_vertex = [uv if uv is not None else default_uv for uv in vert_to_uv]

    # 从纹理中采样颜色
    vertex_colors = sample_texture(texture_path, uv_for_vertex)

    # 写出新的带颜色的 .obj
    write_colored_obj(output_path, vertices, faces, vertex_colors)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5:
        sys.exit(1)

    input_obj = sys.argv[1]
    input_png = sys.argv[2]
    output_obj = sys.argv[3]
    i = sys.argv[4]
    i = int(i)
    convert_to_colored_obj(input_obj, input_png, output_obj)
    print(f"输出文件已保存到: {output_obj}")

