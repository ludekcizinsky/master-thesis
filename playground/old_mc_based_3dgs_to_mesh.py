def _split_person_ids(state: Dict[str, torch.Tensor]) -> np.ndarray:
    # Prefer explicit person_ids, fall back to counts, else assume a single person.
    if "person_ids" in state:
        return state["person_ids"].detach().cpu().numpy().astype(np.int32)

    counts = state.get("person_gaussian_counts", None)
    if counts is not None:
        counts_np = counts.detach().cpu().numpy().astype(np.int32).tolist()
        ids = []
        for pid, c in enumerate(counts_np):
            ids.extend([pid] * int(c))
        return np.asarray(ids, dtype=np.int32)

    return np.zeros((state["xyz"].shape[0],), dtype=np.int32)


def _prepare_gaussians(
    state: Dict[str, torch.Tensor],
    mask: np.ndarray,
    *,
    sigma_scale: float,
    min_sigma: float,
    max_sigma: Optional[float],
    min_opacity: float,
    max_gaussians: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Extract per-person arrays and normalize opacity/scales.
    xyz = state["xyz"].detach().cpu().numpy().astype(np.float32)[mask]
    scaling = state["scaling"].detach().cpu().numpy().astype(np.float32)[mask]
    opacity = state["opacity"].detach().cpu().numpy().astype(np.float32)[mask]

    if opacity.ndim == 2 and opacity.shape[-1] == 1:
        opacity = opacity[:, 0]
    opacity = np.clip(opacity, 0.0, 1.0)

    if max_gaussians is not None and xyz.shape[0] > max_gaussians:
        # Uniform downsample to cap density computation cost.
        rng = np.random.default_rng(seed)
        idx = rng.choice(xyz.shape[0], size=max_gaussians, replace=False)
        xyz = xyz[idx]
        scaling = scaling[idx]
        opacity = opacity[idx]

    # Isotropic sigma from per-axis scales (rotation ignored).
    scales = scaling
    if scales.ndim == 2 and scales.shape[1] == 3:
        sigma = scales.mean(axis=1)
    else:
        raise ValueError(f"Expected scaling shape [N,3], got {scaling.shape}")
    sigma = sigma * float(sigma_scale)
    sigma = np.clip(sigma, min_sigma, max_sigma if max_sigma is not None else np.inf)

    keep = opacity >= min_opacity
    return xyz[keep], sigma[keep], opacity[keep]


def _density_grid_from_gaussians(
    centers: np.ndarray,
    sigmas: np.ndarray,
    weights: np.ndarray,
    grid_size: Tuple[int, int, int],
    padding: float,
    truncation: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    if centers.shape[0] == 0:
        raise ValueError("No gaussians remaining after filtering.")

    # Build a uniform grid over the gaussian bbox with padding.
    min_corner = centers.min(axis=0) - padding
    max_corner = centers.max(axis=0) + padding

    nx, ny, nz = grid_size
    xs = np.linspace(min_corner[0], max_corner[0], nx, dtype=np.float32)
    ys = np.linspace(min_corner[1], max_corner[1], ny, dtype=np.float32)
    zs = np.linspace(min_corner[2], max_corner[2], nz, dtype=np.float32)
    dx = float(xs[1] - xs[0]) if nx > 1 else 1.0
    dy = float(ys[1] - ys[0]) if ny > 1 else 1.0
    dz = float(zs[1] - zs[0]) if nz > 1 else 1.0

    density = np.zeros((nx, ny, nz), dtype=np.float32)

    # Accumulate truncated Gaussian blobs into the grid.
    for mu, sigma, w in zip(centers, sigmas, weights):
        radius = truncation * sigma
        if radius <= 0.0:
            continue
        ix0 = max(0, int(np.floor((mu[0] - radius - min_corner[0]) / dx)))
        ix1 = min(nx - 1, int(np.ceil((mu[0] + radius - min_corner[0]) / dx)))
        iy0 = max(0, int(np.floor((mu[1] - radius - min_corner[1]) / dy)))
        iy1 = min(ny - 1, int(np.ceil((mu[1] + radius - min_corner[1]) / dy)))
        iz0 = max(0, int(np.floor((mu[2] - radius - min_corner[2]) / dz)))
        iz1 = min(nz - 1, int(np.ceil((mu[2] + radius - min_corner[2]) / dz)))

        if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
            continue

        sigma2 = float(sigma * sigma)
        x = xs[ix0 : ix1 + 1] - mu[0]
        y = ys[iy0 : iy1 + 1] - mu[1]
        z = zs[iz0 : iz1 + 1] - mu[2]
        gx = np.exp(-(x * x) / (2.0 * sigma2)).astype(np.float32)
        gy = np.exp(-(y * y) / (2.0 * sigma2)).astype(np.float32)
        gz = np.exp(-(z * z) / (2.0 * sigma2)).astype(np.float32)

        density[ix0 : ix1 + 1, iy0 : iy1 + 1, iz0 : iz1 + 1] += (
            w * gx[:, None, None] * gy[None, :, None] * gz[None, None, :]
        )

    return density, min_corner.astype(np.float32), (dx, dy, dz)


def _extract_mesh(
    density: np.ndarray,
    origin: np.ndarray,
    spacing: Tuple[float, float, float],
    iso_level: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Marching Cubes expects density on a grid; spacing converts to world units.
    verts, faces, _, _ = marching_cubes(density, level=iso_level, spacing=spacing)
    verts = verts + origin[None, :]
    return verts.astype(np.float32), faces.astype(np.int32)


def _write_mesh_npz(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vertices=vertices, faces=faces)


def _format_frame_name(frame_name: str, width: int = 5) -> str:
    if frame_name.isdigit():
        return f"{int(frame_name):0{width}d}"
    return frame_name


def get_meshes_from_3dgs(
    state: Dict[str, torch.Tensor],
    output_dir: Path,
    frame_name: str,
    cfg: MeshConfig,
    *,
    overwrite: bool = False,
    write_meshes: bool = True,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert posed 3DGS to per-person meshes. Optionally write mesh npz files.
    """
    frame_name_fmt = _format_frame_name(frame_name)
    person_ids = _split_person_ids(state)
    grid_size = (int(cfg.grid_size), int(cfg.grid_size), int(cfg.grid_size))
    results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # Convert each person independently to keep per-person mesh outputs.
    unique_persons = np.unique(person_ids)
    for pid in unique_persons:
        out_path = output_dir / f"mesh-f{frame_name_fmt}-p{int(pid)}.npz"
        if write_meshes and out_path.exists() and not overwrite:
            try:
                with np.load(out_path) as npz:
                    vertices = npz["vertices"].astype(np.float32)
                    faces = npz["faces"].astype(np.int32)
                results[int(pid)] = (vertices, faces)
            except Exception:
                pass
            continue

        mask = person_ids == pid
        centers, sigmas, weights = _prepare_gaussians(
            state,
            mask,
            sigma_scale=cfg.sigma_scale,
            min_sigma=cfg.min_sigma,
            max_sigma=cfg.max_sigma,
            min_opacity=cfg.min_opacity,
            max_gaussians=cfg.max_gaussians,
            seed=cfg.seed,
        )
        if centers.shape[0] == 0:
            continue

        density, origin, spacing = _density_grid_from_gaussians(
            centers,
            sigmas,
            weights,
            grid_size=grid_size,
            padding=cfg.padding,
            truncation=cfg.truncation,
        )
        if cfg.iso_percentile is not None:
            # Derive iso-level from non-zero densities to adapt to scale.
            positive = density[density > 0]
            if positive.size == 0:
                continue
            iso_level = float(np.percentile(positive, cfg.iso_percentile))
        else:
            iso_level = cfg.iso_level

        try:
            vertices, faces = _extract_mesh(density, origin, spacing, iso_level)
        except ValueError:
            continue


        if write_meshes:
            _write_mesh_npz(out_path, vertices, faces)
        results[int(pid)] = (vertices, faces)

    return results


@dataclass
class MeshConfig:

    grid_size: int = 96
    padding: float = 0.05
    truncation: float = 3.0
    iso_level: float = 0.05
    iso_percentile: Optional[float] = None
    sigma_scale: float = 1.0
    min_sigma: float = 1e-4
    max_sigma: Optional[float] = None
    min_opacity: float = 0.01
    max_gaussians: Optional[int] = None
    seed: int = 0
    tsdf_voxel_size: float = 0.005
    tsdf_sdf_trunc: float = 0.02
    tsdf_depth_trunc: float = 12.0
    tsdf_alpha_thresh: float = 0.3


def mesh_config_from_cfg(cfg) -> MeshConfig:
    if cfg is None:
        return MeshConfig()

    if isinstance(cfg, dict):
        cfg_dict = cfg
    else:
        try:
            cfg_dict = {k: cfg.get(k) for k in cfg.keys()}
        except Exception:
            cfg_dict = {}

    kwargs = {}
    for field in fields(MeshConfig):
        if field.name in cfg_dict and cfg_dict[field.name] is not None:
            kwargs[field.name] = cfg_dict[field.name]
    return MeshConfig(**kwargs)

