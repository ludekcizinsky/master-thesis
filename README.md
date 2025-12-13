## Relatively Fast and Robust 4D Reconstruction of (MultiPle) Humans from Monocular Video

YouTube is full of videss of people dancing, doing sports, or just walking around. Most of the times, these videos capture not just a single person, but multiple people interacting with each other. Being able to reconstruct such multi-person scenes in 3D + model temporal dynamics (4D) from just a single camera would open up many exciting applications such as:
1. VFX for movies and games which would no longer require expensive multi-camera setups
2. Sports replays: imagine being able to see a 3D replay of a soccer match from any angle you want, even from the perspective of a player
3. Physical AI (markeless motion capture): having access to highly accurate estimates of people's joint positions and body shapes in 3D over time from just video could enable collection of large-scale motion datasets for humanoid robots (e.g. retarget human motion to robots)



## Preprocessing limitations and notes:
- it can happen that sam3 actually fails to detect any humans in the scene, so here I would also need to check if everything went fine.
- it can happen that sam3 will fail to detect certain human for a subset of the frames, so be aware of that
- manual inspection needed at this point and making sure that mask track ids match motion track ids.
- another todo is to pick a frame index for each person track to be used as reference frame during inference.
- I need to ensure I am running over all humans detected in the scene.