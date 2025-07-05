
import sys
import yaml
import numpy as np
import trimesh

i=sys.argv[1]
i=int(i)
mesh_path_idx=f'./results/trans_mesh/Name/Name.obj'



mesh1 = trimesh.load(f"./results/trans_mesh/Name/prior_mesh/{i+1}.obj")  # 原始 mesh
mesh2 = trimesh.load(f"./results/trans_mesh/Name/Name.obj")  # 提取后的 mesh

# 平移和缩放 mesh2 使其和 mesh1 对齐
mesh2.apply_translation(-mesh2.centroid)
scale_factor = mesh1.extents.max() / mesh2.extents.max()
mesh2.apply_scale(scale_factor)
mesh2.apply_translation(mesh1.centroid)

# 保存对齐后的 mesh2 再用来初始化 3DGS
mesh2.export(f"./results/trans_mesh/Name/Name.obj")

with open('./configs/config.yaml', 'r') as file:
    params = yaml.safe_load(file) or {}

# 更新参数
params.update({'param1': str(mesh_path_idx)})

# 写入最终更新后的参数
with open('./configs/config.yaml', 'w') as file:
    yaml.dump(params, file)

