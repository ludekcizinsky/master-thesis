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

```bash
bash preprocess/custom/mp4_to_frames.sh
```

This script will:
1. extract frames from the input mp4 videos
2. create a new scene directory and save the frames into the common format

See the script and add your own videos as needed.

### Infering more features from the monocular frames

After running the above scripts, we should have for each scene a directory with monocular video frames in the expected format. This pipeline copies this data into the preprocessing input directory, and then runs the preprocessing script to infer more features from the monocular frames. These data are then used for training.

Schedule the first part of the preprocessing as follows:

```bash
bash preprocess/custom/get_trn_data.sh part1
```

This script will schedule in parallel the preprocessing for all scenes listed in the configuration file. For each scene, we will then:
1. estimate the smplx poses along with the camera parameters
2. estimate the human masks using sam3
3. estimate the depth maps using depth anything 3

Once the preprocessing is done, then for each scene, do the following manual checks and fixes. Check the human motion dir, and make sure that
1. the number of tracks is correct and delete any unwanted tracks
2. if the mask and pose estimation went smoothly, and there are not any mismatches in the number of tracks etc. you can skip this step. Else, after the cleanup, run the check h3r script to make sure that the data in human motion dir is correct, this script will autiomatically generate skip frames csv as well

```bash
bash  preprocess/custom/check_h3r_output.sh --scene_dir <path_to_scene_dir>
```

3. if you are preprocessing hi4d, then check that the gt segmentation matches sam3 masks tracks and motion tracks. If there is no gt segmentation, then make sure that masked tracks match motion tracks


Once everything is checked, we proceed to schedule the second part of the preprocessing:

```bash
bash preprocess/custom/get_trn_data.sh part2
```

This will infer for each human in the scene its canonical 3dgs representation. Finally, run the reformat script to get the data into the format expected by the trainer:

```bash
bash preprocess/custom/reformat.sh
```

This step does the following:
1. reformat the images, masks, depth maps and cameras into the expected format
2. reformat the h3r data into the expected format: merge all human tracks for each frame into a single file, and ensure that smplx translation and global rotation are in canonical to world format
3. (optional) align the world frames of the estimated smplx poses with the ground truth smplx poses, if available
4. convert smplx to smpl format (this is important for eval since gt has only smpl)


### Saving the state to Hugging Face

To avoid that the preprocessed data is lost, we can save the preprocessed data to Hugging Face. This is done as follows:

```bash
cd /scratch/izar/cizinsky/thesis/preprocessing
hf upload-large-folder ludekcizinsky/thesis-data . --repo-type=dataset --no-private --exclude "misc/*"
```

## How to know that everything went fine?

It is very important to ensure that before we start training and evaluation, the preprocessed data is correct and consistent. Below, I therefore list a few checks that need to be done to ensure that everything is fine.

### Check that we can render smplx meshes from the estimated smplx parameters and cameras

```bash
bash preprocess/vis/check_render.sh mmm_lift 0 
```

This should run for each frame and produce masked images overlaid with the rendered smplx meshes. Check that the smplx meshes align well with the humans in the scene.

### Check the scene in 3D

```bash
bash preprocess/vis/check_scene_in_3d.sh mmm_lift 0 false
```

This will open a 3D viewer where you can see visualizations of:
- the estimated smplx meshes (and smpl if available)
- the estimated cameras
- posed 3dgs (you must set the last argument to true to see these - note that this can be slow to load)

Then you can check that everything is aligned well in 3D.

## Limitations
- it can happen that sam3 actually fails to detect any humans in the scene, so here I would also need to check if everything went fine.
- it can happen that sam3 will fail to detect certain human for a subset of the frames, so be aware of that
- manual inspection needed at this point and making sure that mask track ids match motion track ids as well gt track ids
- another todo is to pick a frame index for each person track to be used as reference frame during inference.
- I need to ensure I am running over all humans detected in the scene.
