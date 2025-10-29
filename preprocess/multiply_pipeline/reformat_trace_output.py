from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence

import numpy as np
import os

CONTACT_COLORS = [[[0.412,0.663,1.0,1.0], [1.0, 0.412, 0.514, 1.0]], [[1.0,0.749,0.412,1.0], [1.0, 0.412, 0.514, 1.0]], [[0.412,1.0,0.663,1.0], [1.0, 0.412, 0.514, 1.0]], [[0.412,0.412,0.663,1.0], [1.0, 0.412, 0.514, 1.0]], [[0.412,0.0,0.0,1.0], [1.0, 0.412, 0.514, 1.0]], [[0.0,0.0,0.663,1.0], [1.0, 0.412, 0.514, 1.0]],[[0.0,0.412,0.0,1.0], [1.0, 0.412, 0.514, 1.0]],[[1.0,0.0,0.0,1.0], [1.0, 0.412, 0.514, 1.0]],[[0.0,1.0,0.0,1.0], [0.0, 0.0, 1.0, 1.0]], [[0.0,0.0,1.0,1.0], [1.0, 0.412, 0.514, 1.0]]]

def process_idx(reorganize_idx, vids=None):
    # reorganize_idx = reorganize_idx.cpu().numpy()
    used_org_inds = np.unique(reorganize_idx)
    per_img_inds = [np.where(reorganize_idx==org_idx)[0] for org_idx in used_org_inds]

    return used_org_inds, per_img_inds

def main(args):

    # render SMPL
    trace_output = f"{args.output_folder}/trace_results/{args.seq}/frames.npz"
    result = np.load(trace_output, allow_pickle=True)
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")
    used_org_inds, per_img_person_inds = process_idx(result['outputs'][()]['reorganize_idx'])
     
    track_ids = result['outputs'][()]['track_ids']
    unique_id = np.unique(track_ids)
    per_img_inds = [np.where(track_ids==id)[0] for id in unique_id]

    theta_np = np.zeros((len(unique_id), len(used_org_inds), 72))
    beta_np = np.zeros((len(unique_id), len(used_org_inds), 10))
    cam_np = np.zeros((len(unique_id), len(used_org_inds), 3))
    j3d_np = np.zeros((len(unique_id), len(used_org_inds), 44, 3))
    pj2d_org_np = np.zeros((len(unique_id), len(used_org_inds), 44, 2))
    verts_np = np.zeros((len(unique_id), len(used_org_inds), 6890, 3))
    for org_ind, img_inds in zip(used_org_inds, per_img_person_inds):
        for img_ind_i in img_inds:
            track_id = track_ids[img_ind_i] -1
            theta_np[track_id, org_ind] = result['outputs'][()]['smpl_thetas'][img_ind_i]
            beta_np[track_id, org_ind] = result['outputs'][()]['smpl_betas'][img_ind_i]
            cam_np[track_id, org_ind] = result['outputs'][()]['cam_trans'][img_ind_i]
            j3d_np[track_id, org_ind] = result['outputs'][()]['j3d'][img_ind_i]
            pj2d_org_np[track_id, org_ind] = result['outputs'][()]['pj2d_org'][img_ind_i]
    
    theta_list = []
    beta_list = []
    cam_list = []
    j3d_list = []
    pj2d_org_list = []
    verts_list = []

    for i, person_id_list in enumerate(per_img_inds):

        theta = result['outputs'][()]['smpl_thetas'][person_id_list]
        beta = result['outputs'][()]['smpl_betas'][person_id_list]
        world_trans = result['outputs'][()]['world_trans'][person_id_list]
        world_rotation = result['outputs'][()]['world_global_rots'][person_id_list]
        cam_trans = result['outputs'][()]['cam_trans'][person_id_list]
        j3d = result['outputs'][()]['j3d'][person_id_list]
        pj2d_org = result['outputs'][()]['pj2d_org'][person_id_list]
        theta_list.append(theta)
        beta_list.append(beta)
        cam_list.append(cam_trans)
        j3d_list.append(j3d)
        pj2d_org_list.append(pj2d_org)

        da_pose = np.zeros_like(theta_np[i, :,3:])
        da_pose[:, 2] = np.pi / 6
        da_pose[:, 5] = - np.pi / 6
        smpl_seq = SMPLSequence(poses_body = theta_np[i, :,3:],
                                smpl_layer = smpl_layer,
                                poses_root = theta_np[i, :, :3],
                                betas = beta_np[i, :,:],
                                trans = cam_np[i, :, :])
        verts_list.append(smpl_seq.vertices)
        verts_np[i] = smpl_seq.vertices
        
        smpl_seq.mesh_seq.vertex_colors = np.array(CONTACT_COLORS[i])[np.zeros(6890,dtype=np.int32)][np.newaxis,...].repeat(world_trans.shape[0],axis=0)
        smpl_seq.name = "smpl" + str(i)
        smpl_seq.mesh_seq.material.diffuse = 1.0
        smpl_seq.mesh_seq.material.ambient = 0.1
    
    save_result = {
        "joints": j3d_np,
        "pj2d_org": pj2d_org_np,
        "cam_trans": cam_np,
        "smpl_thetas": theta_np,
        "smpl_betas": beta_np,
        "verts": verts_np
    }

    os.makedirs(f"{args.output_folder}/raw_data/{args.seq}/trace", exist_ok=True)
    np.savez(f"{args.output_folder}/raw_data/{args.seq}/trace/{args.seq}.npz", results=save_result)
    print(f"Reformatted trace results saved to {args.output_folder}/raw_data/{args.seq}/trace/{args.seq}.npz")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, default="seq_name", help='seq name')
    parser.add_argument("--output_folder", type=str, default="/scratch/izar/cizinsky/multiply-output/preprocessing", help="output folder")
    main(parser.parse_args())
