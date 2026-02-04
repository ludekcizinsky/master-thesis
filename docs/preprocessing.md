## How this works

In this section, we describe how to prepare the data for training and evaluation. 


### Preparing GT evaluation data

In this section, we describe how to prepare the ground truth evaluation data for each dataset. This step is important to ensure that during evaluation, we can compare our reconstructed results with the ground truth data in a consistent manner.

#### Hi4D dataset

```bash
bash preprocess/eval/hi4d/reformat.sh
``` 

This script will:
1. reformat static cameras into the common format
2. reformat meshes into the common format
3. convert smpl to smplx format

#### MMM

```bash
bash preprocess/eval/mmm/reformat.sh
``` 

This script will:
1. downscale the images to the expected resolution
2. reformat the dynamic camera into the common format and adjust intrinsics accordingly to the new image resolution
3. reformat meshes into the common format

### Preparing training data from monocular videos

The preprocessing pipeline expects a scene directory with `images/camera_id/` containing the frames for the input camera in the `jpg` format using 6 digits for the frame number (e.g., `000001.jpg`). We can initialise this scene directory by running the following script:

```bash
python preprocess/train/init_scene_dir.py \
  --video-path /path/to/video_or_frames \
  --seq-name my_sequence \
  --cam-id 0 \
  --output-dir /path/to/output_root
```

This script will:
1. extract frames from the input video (if a video path is provided) and save them in the expected format. If the input path is already a directory containing frames, it will simply copy them to the output directory in the expected format (6-digit zero-padded filenames).

Then, we can run the off-the-shelf methods to infer the initial scene reconstruction as follows:

```bash
python preprocess/train/infer_initial_scene.py \
  --scene-dir /path/to/initialized_scene_dir \
  --ref-frame-idx 0 
```

This script will:
1. run PromptHMR with SAM3 to infer the invidual human masks, SMPLX and camera parameters in the world coordinate system 
2. run Depth Anything to infer the depth maps for each frame 
3. run Large Human Model (LHM) to obtain initial canonical 3dgs representation of the humans in the scene, it will use the `ref-frame-idx` to select the reference frame to infer the canonical 3dgs representation 
4. run virtual camera generation to create additional static cameras around the input camera to improve the coverage of the scene during training, these will be later used to render additional training views and improve these using DiFix

### Visualising the preprocessed data

To see the results of the preprocessing of both the GT evaluation data and the training data, we can use the following script:

```bash
python preprocess/vis/check_scene_in_3d.py \
  --scenes-dir /path/to/scenes_dir \
  --src_cam_id 0 \
  --frame-idx-range 0 10
```