import os
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video

from diffusers import DiffusionPipeline
import torch
import rembg
import requests
import json
import yaml

# Configuration and Load parameters

parser = argparse.ArgumentParser()
parser.add_argument('text', type=str, help='Path to input text.')
parser.add_argument('name', type=str, help='Output name.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output path.')
parser.add_argument('--MV_diffusion_steps', type=int, default=75, help='Multi_View Denoising steps.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of Multi-view.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_omp', action='store_true', help='Export obj, mtl, png.')
parser.add_argument('--seed', type=int, default=1, help='Seed for random number generator.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale for generated object.')
args = parser.parse_args()
seed_everything(args.seed)

config = OmegaConf.load("configs/reconstruction.yaml")
config_name = "instant-mesh-large"
model_config = config.model_config
infer_config = config.infer_config



device = torch.device('cuda')


pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)


if os.path.exists(infer_config.unet_path):
    unet_ckpt_path = infer_config.unet_path
else:
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device)

model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=False)

model = model.to(device)
model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()


image_path = "./results/image"
image2_path = "./results/MVimage"
prior_path = "./results/prior_mesh"
gs_path = "./results/gs"
extraction_path = "./results/mesh_extraction"
trans_mesh="./results/trans_mesh"
os.makedirs(image_path, exist_ok=True)
os.makedirs(image2_path, exist_ok=True)
os.makedirs(prior_path, exist_ok=True)
os.makedirs(gs_path, exist_ok=True)
os.makedirs(extraction_path, exist_ok=True)
os.makedirs(trans_mesh, exist_ok=True)

name=args.name
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.to("cuda")
prompt=args.text
combined_text=f"a ((full-body:2)) shot of a ((single:2)) {prompt}, isolated on gray background, 4k, highly detailed"
input_files=[]

image = pipe(combined_text, num_inference_steps=50, guidance_scale=7.5).images[0]
print(combined_text)
img_path="./results/image/"+name+".png"

with open('./configs/config.yaml', 'r') as file:
    params = yaml.safe_load(file) or {}

# 更新参数
params.update({'param2': str(img_path)})

# 写入更新后的参数
with open('./configs/config.yaml', 'w') as file:
    yaml.dump(params, file)


input_files.append(img_path)
image.save(img_path)  # 保存图像名
del pipe



print("Image-to-Multi_View Generation")

rembg_session = None if args.no_rembg else rembg.new_session()

outputs = []
for idx, image_file in enumerate(input_files):
    name = os.path.basename(image_file).split('.')[0]
    print(f'[{idx+1}/{len(input_files)}] Imagining {name} ...')

    input_image = Image.open(image_file)
    if not args.no_rembg:
         input_image = rembg.remove(input_image)
    # sampling
    output_image = pipeline(
        input_image,
        num_inference_steps=args.MV_diffusion_steps,
    ).images[0]
    output_image.save(os.path.join(image2_path, f'{name}.png'))


    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)
    input_image=np.asarray(output_image, dtype=np.float32) / 255.0
    input_image = torch.from_numpy(input_image).permute(2, 0, 1).contiguous().float()
    input_image = input_image.unsqueeze(0)
    outputs.append({'name': name, 'images': images,"name2":name,"main_image":input_image})


del pipeline

# Text-Infused Sparse-View 3D Reconstruction

print("3D Reconstruction")

input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device)
chunk_size = 20

for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    Main_image= sample['main_image'].unsqueeze(0).to(device)

    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)
    Main_image = v2.functional.resize(Main_image, 320, interpolation=3, antialias=True).clamp(0, 1)

    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5]).long().to(device)
        images = images[:, indices]
        input_cameras = input_cameras[:, indices]

    with torch.no_grad():
        # get triplane

        planes = model.forward_planes(images, input_cameras)

        # get mesh
        mesh_path_idx = os.path.join(prior_path, f'{name}.obj')

        with open('./configs/config.yaml', 'r') as file:
            params = yaml.safe_load(file) or {}

        # 更新参数
        params.update({'param1': str(mesh_path_idx)})

        # 写入最终更新后的参数
        with open('./configs/config.yaml', 'w') as file:
            yaml.dump(params, file)

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_omp,
            **infer_config,
        )
        if args.export_omp:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)


        def rotate_z(vertices, angle):
            """绕y轴旋转顶点"""
            angle_rad = np.radians(angle)
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
            return np.dot(vertices, rotation_matrix)
        def rotate_x(vertices, angle):
            """绕x轴旋转顶点"""
            angle_rad = np.radians(angle)
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
            return np.dot(vertices, rotation_matrix)



        def process_obj_file(input_file, output_file, angle,angle_z):
            """处理OBJ文件，绕x轴旋转顶点"""
            with open(input_file, 'r') as f:
                lines = f.readlines()

            with open(output_file, 'w') as f:
                for line in lines:
                    if line.startswith('v '):  # 顶点数据行
                        # 提取顶点坐标和颜色
                        vertex = np.fromstring(line[2:-1], dtype=float, sep=' ')
                        # 分割顶点坐标和颜色
                        if len(vertex) >= 6:  # 确保有足够的数据
                            position = vertex[:3]  # 顶点位置
                            color = vertex[3:6]  # 顶点颜色
                        else:
                            position = vertex[:3]  # 顶点位置
                            color = [1.0, 1.0, 1.0]  # 默认颜色为白色

                        # 绕x轴旋转顶点位置
                        rotated_position = rotate_x(position, angle)
                        rotated_position = rotate_z(rotated_position, angle_z)
                        # 写入旋转后的顶点位置和颜色
                        f.write('v {} {} {} {} {} {}\n'.format(*rotated_position, *color))
                    else:
                        f.write(line)

        input_obj_file = mesh_path_idx  # 输入OBJ文件路径
        output_obj_file = mesh_path_idx  # 输出OBJ文件路径
        rotation_angle = 180  # 旋转角度
        rotation_angle_z = 90
        process_obj_file(input_obj_file, output_obj_file, rotation_angle,rotation_angle_z)


        def mirror_mesh_x_with_normals_and_fixed_winding(input_file, output_file):
            """
            镜像 OBJ 网格文件（沿 X 轴），修复法线和面片绕序以保持正确光照和朝向。
            """
            with open(input_file, 'r') as f:
                lines = f.readlines()

            with open(output_file, 'w') as f:
                for line in lines:
                    if line.startswith("v "):  # 顶点坐标
                        vertex = np.fromstring(line[2:], dtype=float, sep=' ')
                        if len(vertex) >= 3:
                            vertex[0] = -vertex[0]  # 镜像 X 坐标
                        f.write('v {} {} {}{}\n'.format(
                            *vertex[:3],
                            ' ' + ' '.join(map(str, vertex[3:])) if len(vertex) > 3 else ''
                        ))
                    elif line.startswith("vn "):  # 顶点法线
                        normal = np.fromstring(line[3:], dtype=float, sep=' ')
                        if len(normal) >= 3:
                            normal[0] = -normal[0]  # 镜像 X 法线
                        f.write('vn {} {} {}\n'.format(*normal[:3]))
                    elif line.startswith("f "):  # 面片（顶点顺序翻转）
                        parts = line.strip().split()
                        if len(parts) == 4:
                            v1, v2, v3 = parts[1], parts[2], parts[3]
                            f.write(f"f {v1} {v3} {v2}\n")  # 翻转 v2 和 v3
                        else:
                            f.write(line)
                    else:
                        f.write(line)


        input_obj_file = mesh_path_idx
        output_obj_file = mesh_path_idx
        mirror_mesh_x_with_normals_and_fixed_winding(input_obj_file, output_obj_file)

        print(f"3D Mesh saved to {mesh_path_idx}")
