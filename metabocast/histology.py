"""Lightweight H&E feature extraction for MetaboCast optional inputs.

The public predictor remains transcriptome-only by default. This helper adds a
small, auditable per-spot histology feature block to `adata.obsm`, which can be
used with `MetaboCast(extra_key="histology")` for ablation experiments.
"""
import numpy as np


FEATURE_NAMES = [
    "rgb_mean_r", "rgb_mean_g", "rgb_mean_b",
    "rgb_std_r", "rgb_std_g", "rgb_std_b",
    "od_mean_r", "od_mean_g", "od_mean_b",
    "gray_mean", "gray_std", "edge_mean", "tissue_fraction",
]


def _dense_image(img):
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("Histology image must be an RGB array with shape (height, width, 3).")
    arr = arr[:, :, :3].astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _resolve_library_id(adata, library_id=None, obs_library_key="orig.ident"):
    spatial = adata.uns.get("spatial", {})
    if not spatial:
        raise ValueError("adata.uns['spatial'] is missing; cannot locate embedded histology images.")
    if library_id is not None:
        if library_id not in spatial:
            raise ValueError(f"library_id={library_id!r} not found in adata.uns['spatial'].")
        return library_id
    if obs_library_key in adata.obs:
        vals = [str(v) for v in adata.obs[obs_library_key].unique()]
        hits = [v for v in vals if v in spatial]
        if len(hits) == 1:
            return hits[0]
    if len(spatial) == 1:
        return next(iter(spatial))
    raise ValueError(
        "Could not infer the matching Visium library image. Pass library_id=... explicitly; "
        f"available libraries are {list(spatial.keys())}."
    )


def _image_and_scale(adata, library_id, image_key="hires"):
    entry = adata.uns["spatial"][library_id]
    images = entry.get("images", {})
    if image_key not in images:
        if image_key == "hires" and "lowres" in images:
            image_key = "lowres"
        else:
            raise ValueError(f"No {image_key!r} image for library_id={library_id!r}.")
    img = _dense_image(images[image_key])
    scalefactors = entry.get("scalefactors", {})
    scale_key = "tissue_hires_scalef" if image_key == "hires" else "tissue_lowres_scalef"
    scale = scalefactors.get(scale_key)
    if scale is None:
        coords = np.asarray(adata.obsm["spatial"], float)
        # Last-resort inference for old objects that kept an image but dropped scale metadata.
        scale = min(img.shape[1] / max(coords[:, 0].max(), 1.0), img.shape[0] / max(coords[:, 1].max(), 1.0))
    spot_diam = float(scalefactors.get("spot_diameter_fullres", 15.0)) * float(scale)
    return img, float(scale), max(3, int(round(0.5 * spot_diam)))


def _patch_features(patch):
    if patch.size == 0:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
    rgb_mean = patch.mean(axis=(0, 1))
    rgb_std = patch.std(axis=(0, 1))
    od = -np.log(np.clip(patch, 1e-3, 1.0))
    od_mean = od.mean(axis=(0, 1))
    gray = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
    if gray.shape[0] > 1 and gray.shape[1] > 1:
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        edge = 0.5 * (np.abs(gx).mean() + np.abs(gy).mean())
    else:
        edge = 0.0
    tissue = np.mean((gray < 0.92) & ((patch.max(axis=2) - patch.min(axis=2)) > 0.03))
    return np.asarray([*rgb_mean, *rgb_std, *od_mean, gray.mean(), gray.std(), edge, tissue], dtype=np.float32)


def add_histology_features(
    adata,
    key_added="histology",
    library_id=None,
    obs_library_key="orig.ident",
    image_key="hires",
    patch_radius_scale=1.0,
    standardize=True,
):
    """Attach cheap per-spot H&E features to `adata.obsm[key_added]`.

    Coordinates are assumed to be Space Ranger full-resolution pixel coordinates
    in `adata.obsm['spatial']`; embedded Visium scalefactors map them to the
    selected `hires` or `lowres` image. The feature vector is deliberately small:
    RGB means/stds, optical-density means, grayscale/edge summaries and tissue
    fraction. Use `MetaboCast(extra_key=key_added)` to include them.
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is required for histology feature extraction.")
    lib = _resolve_library_id(adata, library_id, obs_library_key)
    img, scale, base_radius = _image_and_scale(adata, lib, image_key)
    coords = np.asarray(adata.obsm["spatial"], float) * scale
    radius = max(2, int(round(base_radius * float(patch_radius_scale))))
    h, w = img.shape[:2]
    feats = np.zeros((coords.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    for i, (x, y) in enumerate(coords):
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        feats[i] = _patch_features(img[y0:y1, x0:x1])
    if np.isnan(feats).any():
        means = np.nanmean(feats, axis=0)
        inds = np.where(np.isnan(feats))
        feats[inds] = np.take(means, inds[1])
    if standardize:
        feats = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-8)
        feats = feats.astype(np.float32)
    adata.obsm[key_added] = feats
    adata.uns[key_added + "_feature_names"] = list(FEATURE_NAMES)
    adata.uns[key_added + "_source"] = {
        "library_id": lib,
        "image_key": image_key,
        "coordinate_scale": scale,
        "patch_radius_pixels": radius,
        "standardized_per_section": bool(standardize),
    }
    return adata
