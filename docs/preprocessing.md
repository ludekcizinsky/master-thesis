## How this works

During preprocessing, we not only need to extract new features from the monocular video frames, but also need to ensure that the data is in the common format expected by the trainer. Further, we also need to ensure that the preprocessed dataset can be compared with the corresponding ground truth dataset.

*Note: as of this moment, the preprocessing pipeline is semi-automatic and requires some manual intervention to ensure that the results are correct. This will be hopefully improved in the future.*

### Preparing inputs

As already mentioned, the preprocessing pipeline expects monocular video frames as input. Indeed, we might also have access to some other ground truth data against which we would like to compare our results. For this reason, I have developed three types of input scene preparation scripts that will prepare the input data into the expected format.

#### Hi4D dataset

```bash
bash preprocess/hi4d/reformat.sh
``` 

This script will:
1. reformat static cameras into the common format
2. reformat meshes into the common format

#### MMM

```bash
bash preprocess/mmm/reformat.sh
``` 

This script will:
1. downscale the images to the expected resolution
2. reformat the dynamic camera into the common format and adjust intrinsics accordingly to the new image resolution
3. reformat meshes into the common format

#### In-the-wild videos

```bash
bash preprocess/custom/mp4_to_frames.sh
```

This script will:
1. extract frames from the input mp4 videos
2. create a new scene directory and save the frames into the common format

See the script and add your own videos as needed.

### Infering more features from the monocular frames

After running the above scripts, we should have for each scene a directory with monocular video frames in the expected format. This pipeline copies this data into the preprocessing input directory, and then runs the preprocessing script to infer more features from the monocular frames. These data are then used for training.


Given you have these inputs, proceed as follows:
1. call prerocess.sh with the scene dir, src_cam_id and seq_name, however make sure that the last part (lhm) is commented out for now
2. once the preprocessing is done, go to the output dir and
    a. check the human motion dir, and make sure that 1. the number of tracks is correct and delete any unwanted tracks
    b. run the check h3r script to make sure that the data in human motion dir is correct, this script will autiomatically generate skip frames csv as well
    b. if you are preprocessing hi4d, then check that the gt segmentation matches sam3 masks tracks and motion tracks. If there is no gt segmentation, then make sure that masked tracks match motion tracks
3. once everything is checked, go back to preprocess.sh and uncomment the last part (lhm) and run it again, this will generate canonical 3dgs models for each human
4. finally, run the reformat script to get the data into the format expected by the trainer


### Limitations
- it can happen that sam3 actually fails to detect any humans in the scene, so here I would also need to check if everything went fine.
- it can happen that sam3 will fail to detect certain human for a subset of the frames, so be aware of that
- manual inspection needed at this point and making sure that mask track ids match motion track ids as well gt track ids
- another todo is to pick a frame index for each person track to be used as reference frame during inference.
- I need to ensure I am running over all humans detected in the scene.
