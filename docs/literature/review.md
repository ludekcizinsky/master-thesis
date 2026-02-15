# Literature review

This page contanins collection of related works which I think are relevant to this project. I will try to keep it up to date as I read more papers and find more related works. In each section, I try to list the works in *chronological* order.


## Pose estimation from video

### 25 March / PromptHMR
Main takeaways from reading PromptHMR paper:
- Offline method (the tradeoff) which however outperforms in terms of quality Human3R
- Can work with video, and can output human poses in world coordinates
- It was also trained on hi4d, specifically, the possible point of conflict with my evaluation setup is that pair 17 is in their training data, the remainder of the pairs is in test
- It would nicely fit with sam3 which would provide input masks and tracking as input prompts, this would also allow me to o resolve the issue of having to align sam and human3r tracks
- It does suffer from small interpenetrations but that would be a perfect opportunity for improvement through my method
- They also have an option to pass interaction prompt through which you specific which two persons (only 2 even if there are more people) are interacting in the scene -> this trigger pass of the tokens through a special module in the model

## Fix render diffusion models

### 25 March / DiFix3D
Main takeaways from Difix:
1. In general, to train DiFix you need pair of rendered and ground truth images. To render these views, you need to also have access to some training views. They say that a possible strategy for datasets such as as DL3DV with large enough camera motion (people go around the scene and take video) is to use the mono video as supervision as follows:
    1. Sample every nth frame and use it for training
    2. Use the remainder for novel view synthesis - render and use it as training pair for difix training
2. However, for other datasets such as the one from Mip-nerf that has large overlap between the held out views and training views, this strategy doesn’t to work so well. So for this reason if you have for instance AV capture that has linear camera trajectory, they train on this trajectory and define the novel views as new cam trajectory shifted 1-6 meters horizontally.
3. In addition to make the model more robust, they underfit intentionally the 3d rep - e.g. using only 25% to 75% of the original training data
4. Finally, for the multi cam capture, the data collection becomes trivial since they just use one of the cams as training and the remainder as novel views. 
5. I believe that the model they released uses Dl3DV only, even if they also used the other dataset - internal self driving dataset, these are way different domains then what we are asking the model to do.

### 25 April / FlowR

Main tetakeaways from FlowR:
- Dataset of size 3.6M of render-gt pairs from 10.3k scenes, the majority of data comes from dl3dv and then scannet++
