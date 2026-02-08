### My backlog list of things to do in order to improve this project

1. 3DGS to mesh conversion improvement: 
a. try using more depth cameras for the tsdf fusion -> it should improve water tightness of the mesh, which in turn should fix the low V-IoU problem, and in addition, it should hopefully also add more details to the mesh 
b. try to add some tunable 2dgs like params that would turn our underlying representaion more mesh ready
2. Make a demo of the reconstruction using rerun io
3. Make the preprocessing pipeline fully automated - ideally use SAM3D to get masks and tracks automatically from video -> also probabky would need to add some heuristic to remove tracks that are too short or not humans. Then feed that to PromptHMR to obtain get initial smplx params and camera poses
4. would be nice to use +y is up as the convention for the smplx params
5. Use SAM3D to add reconstruction of objects in the scene
7. Using 2D pose estimator as an extra step in the pipeline to improve initial pose estimates
8. Better Difix model via:
    a.  training on in-domain data (human data)
    b.  better architecture - currently can take only a single image as contidion
9. Better 3dgs initialisation from LHM - currently always use the first frame to predict human, too naive, would be better to use somthing more sophisticated
10. currently, when doing the difix refinement for novel view, i choose the exact same time point 
11. maybe I was all wrong and the alternative approach with 1. mono to multivide video 2. fit 4dgs is way to go. There is a new paper called MV performer that shows sort of this path for scenes with a single human. I think it would be worth testing this pipleine and see how well it extends to multiple humans and possibly more complex environments 
12. End to end model - we know that LHM is strong at getting canonical human representation, but not so good at motion. While Human3R can get us motion, but is not trained to predict cano representation, maybe these two could be combined
13. project website - get isnpired from 
- difix: https://research.nvidia.com/labs/toronto-ai/difix3d/
- https://monst3r-project.github.io/

14. I wonder how well the 3dgs of depth anything 3 would work for our use case
15. clean up the args to the renderer and give shoutout to LHM authors that I have adapted their 3dgs model
16. figure out how to exclude properly the misc folder when uploading the preprocessed data
17. adding LR scheduler to the training pipeline
18. sam3 works in general well, but it still fails in cases where people interact closely, e.g. it assigns part of the person A's hand to person B - e.g. mmm walkdance segmentation results.
19. More practical setting: assume we have access to high quality canonical humans, and therefore, we just need to estimated the motion from the given scene and (optionally) tune the estimated motion. This way, you can offload the detail apperance task to relying on the high quality canonical human, and just focus on getting the motion right. This is much more feasible in Veo's setting where you have very little about the appearance, but pose is much more tractable