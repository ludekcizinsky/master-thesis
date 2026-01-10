## Dataset Structure

In general, our goal is to reconstruct the given scene from a monocular video. Since this is extremelly challenging and illposed task, we need to first infer additional information about the scene, such as depth maps, camera poses, human poses, etc. Therefore, the dataset structure is designed to accommodate not only the raw images but also these auxiliary data. Since different datasets come in different formats, we always reformat these datasets into a common structure for easier processing. Therefore, at runtime, we assume the structure outlined below. This makes things easier not only for training but also for evaluation and visualization.

Each scene is stored in its own directory, with the following structure:

```
scene_dir/
├── all_cameras
├── images
├── seg
├── smpl
├── smplx
├── depths
├── meshes
├—- canon_3dgs_lhm
| skip_frames.csv (optional)
```

In general, across all modalities, we use the same frame naming convention: `000001`, `000002` - zero-padded 6-digit frame indices. Importantly, frames are aligned across all modalities, i.e., frame `000001` in `images` corresponds to frame `000001` in `all_cameras`, `seg`, `smpl`, etc. Due to various reasons, some frames may be missing in certain modalities (e.g., failed human pose estimation). In such cases, the missing frames should simply be skipped. This is why we also provide an optional `skip_frames.csv` file that lists all the missing frames indices on a single line and comma separated, e.g.,

```
skip_frames.csv:
5, 12, 13, 27
```

where frames `000005`, `000012`, `000013`, and `000027` should be skipped. Importantly, this file is optional. If it does not exist, it means all frames are present. In addition, if for the given modality we are missing certain frame, we will still save empty files for those frames to maintain consistency. 

### all_cameras
---

This folder contains per-frame camera parameters for each camera.
Each camera has its own directory, and each frame is saved as a small `.npz`
file with the keys `intrinsics` and `extrinsics`. Extrinsics parameters denote world-to-camera transformations.

```
scene_dir/
├── all_cameras/
│   ├── XX                              # camera id e.g. 4, 16 etc.
│   │   ├── 000001.npz                  # keys: intrinsics, extrinsics
│   │   ├── 000002.npz
```

`intrinsics` is stored as shape `(1, 3, 3)` and `extrinsics` as shape `(1, 3, 4)`.  Note that in the case camera is static, we still store per-frame files for consistency. In addition, the intrinsics parameters indeed account for any image resizing that may have been applied to the original images.

### images
---

This folder contains the RGB images of the sequence.

```
scene_dir/
├── images/
│   ├── XX                              # camera id e.g. 4, 16 etc.
│   │   ├── 000001.jpg                  # RGB images
```

### seg
---

This folder contains per-frame segmentation masks for each camera.
Masks are stored as `.png` images under `img_seg_mask/<cam_id>/all`.

```
scene_dir/
├── seg/
│   ├── img_seg_mask/
│   │   ├── XX                          # camera id e.g. 4, 16 etc.
│   │   │   ├── all/
│   │   │   │   ├── 000001.png          # binary mask
│   │   │   │   ├── 000002.png
```

Masks are 8-bit grayscale images with values `0` (background) and `255`
(foreground), and the resolution matches the corresponding RGB images.

### smpl
---

This folder contains per-frame SMPL outputs for all detected people. Each
frame is stored as an `.npz` file with the first dimension indexing people.

```
scene_dir/
├── smpl/
│   ├── 000001.npz
│   ├── 000002.npz
```

Expected keys (shapes are for `P` people):
- `betas`: `(P, 10)` shape coefficients
- `global_orient`: `(P, 3)` axis-angle root orientation
- `body_pose`: `(P, 23, 3)` axis-angle body joints
- `transl`: `(P, 3)` global translation

The global orientation and translation define the transformation from the SMPL canonical space to the world space.

### smplx
---

This folder contains per-frame SMPL-X outputs for all detected people. Each
frame is stored as an `.npz` file with the first dimension indexing people.

```
scene_dir/
├── smplx/
│   ├── 000001.npz
│   ├── 000002.npz
```

Expected keys (shapes are for `P` people):
- `betas`: `(P, 10)` shape coefficients
- `root_pose`: `(P, 3)` axis-angle root orientation
- `body_pose`: `(P, 21, 3)` axis-angle body joints
- `jaw_pose`: `(P, 3)` axis-angle jaw pose
- `leye_pose`: `(P, 3)` axis-angle left eye pose
- `reye_pose`: `(P, 3)` axis-angle right eye pose
- `lhand_pose`: `(P, 15, 3)` axis-angle left hand pose
- `rhand_pose`: `(P, 15, 3)` axis-angle right hand pose
- `trans`: `(P, 3)` global translation
- `expression`: `(P, 100)` expression coefficients

The global orientation and translation define the transformation from the SMPL canonical space to the world space.

### depths
---

This folder contains per-frame depth maps for each camera, stored as `.npy`
arrays. Each camera has its own directory.

```
scene_dir/
├── depths/
│   ├── XX                              # camera id e.g. 4, 16 etc.
│   │   ├── 000001.npy                  # depth map (H, W)
│   │   ├── 000002.npy
```

Depth maps are stored as `float32` arrays with shape `(H, W)`; the spatial
resolution matches the corresponding RGB images for that camera. Depth values are in meters.


### meshes
---

This folder contains per-frame mesh reconstructions saved as `.obj` files.

```
scene_dir/
├── meshes/
│   ├── 000001.obj
│   ├── 000002.obj
```

Each `.obj` stores the reconstructed scene geometry for the corresponding
frame as standard OBJ data (`v` vertices and `f` faces).

### canon_3dgs_lhm
---

This folder contains canonical 3D Gaussian splat models for each person, as produced by the LHM pipeline.

```
scene_dir/
├── canon_3dgs_lhm/
│   ├── 00/                       # person 0
│   │   ├── human3r_gs.pt
│   │   ├── human3r_input_image.png
│   │   ├── human3r_input_head.png
│   ├── 01/                       # person 1
│   │   └── ...
│   ├── union/                    # combined models across persons
│   │   ├── human3r_gs.pt
```

Notes:
- `00`, `01`, ... are per-person canonical models (one directory per tracked person).
- `*_gs.pt` are torch checkpoints containing the 3DGS parameters (xyz, scaling, rotation, opacity, SH features) in the canonical pose.
- `*_input_image.png` / `*_input_head.png` are reference images used to build the canonical model.
- `union/` stores a combined set of models across all persons 
