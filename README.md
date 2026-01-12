## Relatively Fast and Robust 4D Reconstruction of (MultiPle) Humans from Monocular Video

YouTube is full of videss of people dancing, doing sports, or just walking around. Most of the times, these videos capture not just a single person, but multiple people interacting with each other. Being able to reconstruct such multi-person scenes in 3D + model temporal dynamics (4D) from just a single camera would open up many exciting applications such as:
1. VFX for movies and games which would no longer require expensive multi-camera setups
2. Sports replays: imagine being able to see a 3D replay of a soccer match from any angle you want, even from the perspective of a player
3. Physical AI (markeless motion capture): having access to highly accurate estimates of people's joint positions and body shapes in 3D over time from just video could enable collection of large-scale motion datasets for humanoid robots (e.g. retarget human motion to robots)



### My backlog list of things to do in order to improve this project

1. 3DGS to mesh converstion: try rendering 3dgs from multiple views and using a multi-view stereo method to get a mesh (e.g. what they do in 2dgs paper as baseline). More advanced would be to add the extra 2dgs params to the predicted 3dgs params from LHM and tune them.
2. Make a demo of the reconstruction using rerun io
3. Make the preprocessing pipeline fully automated - ideally use SAM3D to get masks and tracks automatically from video -> also probabky would need to add some heuristic to remove tracks that are too short or not humans. Then feed that to PromptHMR to obtain get initial smplx params and camera poses
4. would be nice to use +y is up as the convention for the smplx params
5. Use SAM3D to add reconstruction of objects in the scene
6. Tuning pose during the 3dgs optimization stage as well
7. Using 2D pose estimator as an extra step in the pipeline to improve initial pose estimates
8. Better Difix model via:
    a.  training on in-domain data (human data)
    b.  better architecture - currently can take only a single image as contidion
9. Better 3dgs initialisation from LHM - currently always use the first frame to predict human, too naive, would be better to use somthing more sophisticated
10. currently, when doing the difix refinement for novel view, i choose the exact same time point 