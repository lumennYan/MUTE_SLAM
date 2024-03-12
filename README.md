# MUTE-SLAM: Real-Time Neural SLAM with Multiple Tri-plane Hash Representations
## Installation
First you can create the environment and install the necessary dependencies. You can easily achieve this by using anaconda.
```bash
# Create the environment
conda create -n mute_slam python=3.7
conda activate mute_slam
# Install pytorch according to your cuda version
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
# Install the dependencies
pip install -r requirements.txt
```
We use the encoding from [torch-ngp](https://github.com/ashawkey/torch-ngp) by default, you can also use [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) as another option.
To install tiny-cuda-nn, run:
```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
cd tiny-cuda-nn/bindings/torch
python setup.py install
```
## Dataset
### Replica
Download the data from Replica dataset as below:
```bash
bash scripts/download_replica.sh
```
### ScanNet
Follow the instruction on [ScanNet](http://www.scan-net.org/), and extract color/depth frames from the `.sens` file using this [code](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).
### TUM RGB-D
Download the data from TUM RGB-D as below:
```bash
bash scripts/download_tum.sh
```
## Run
You can run MUTE-SLAM with:
```bash
python run.py configs/{Dataset}/{scene}.yaml
```
## Evaluation
### Tracking
To evaluate the tracking result, run:
```bash
python src/tools/eval_ate.py configs/{Dataset}/{scene}.yaml
```
### Reconstruction
To evaluate the reconstruction results, download the ground truth Replica meshes first:
```bash
bash scripts/download_replica_mesh.sh
```
Then cull the unseen and occluded regions from the ground truth meshes:
```bash
GT_MESH=cull_replica_mesh/{scene}.ply
python src/tools/cull_mesh.py configs/Replica/{scene}.yaml --input_mesh $GT_MESH
```
Finally, run the code below to evaluate the reconstructed mesh:
```bash
OUTPUT_FOLDER=output/Replica/{scene}
GT_MESH=cull_replica_mesh/{scene}_culled.ply
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec_culled.ply --gt_mesh $GT_MESH -2d -3d
```
