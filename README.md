#  VisionDreamer: High-Fidelity Text-to-3D Generation via Mesh-Guided 3D Gaussian Splatting

## [Project Page](https://sites.google.com/view/visiondreamer)
# :wrench: Install
    git https://github.com/OpenMICG/VisionDreamer.git

    cd VisionDreamer
    
    conda create -n VisionDreamer python=3.9

    conda activate VisionDreamer

    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

    pip install ninja
    
    pip install ./gaussiansplatting/submodules/diff-gaussian-rasterization
    
    pip install ./gaussiansplatting/submodules/simple-knn
    
    pip install -r requirements.txt
    
    pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch

    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
    pip install ./diff-gaussian-rasterization

# :art:  How to Use
Mesh Prior Generation.
   
    python mesh_generation.py "Input Text" Name

3DGS Model Initialization and Base Optimization.

    python optimization.py system.prompt_processor.prompt="Input Text"

Mesh Extraction From Base Optimized 3DGS model.
   
    python mesh_extraction.py save_path="./results/mesh_extraction/Name"

Mesh-Driven 3DGS Refinement.

    python transmesh.py ./results/mesh_extraction/Name/Name_mesh.obj ./results/mesh_extraction/Name/Name_mesh_albedo.png ./results/trans_mesh/Name/Name.obj Name
    python alignment.py Name
    python refinement.py system.prompt_processor.prompt="Input Text"

# :pray: Anckowledgement
This repo is based on [InstantMesh](https://github.com/TencentARC/InstantMesh), [GaussianDreamer](https://github.com/hustvl/GaussianDreamer), [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian), [threestudio](https://github.com/threestudio-project/threestudio), [zero123plus](https://github.com/SUDO-AI-3D/zero123plus), and [3DGS](https://github.com/graphdeco-inria/gaussian-splatting).  We would like to express our gratitude for their outstanding work.
