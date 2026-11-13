"""Per-frame GeoJSON export for synthetic dataset masks.

The contour-extraction pieces (`mask_to_contours`, `_simplify_contour`,
`clean_and_close_contours`, `_contour_to_geojson_coords`) are vendored from
`Tissue_pseudochannel_gen/src/analysis/export.py` (a separate repo/env) -- pure numpy+skimage,
no QuPath-cluster-color or MCMICRO-batch machinery needed here.
"""

import numpy as np
from skimage import measure


def mask_to_contours(mask, simplify_tolerance=1.0):
    """Extract polygon contours from a segmentation mask.

    Args:
        mask: 2D integer array with cell labels (0=background).
        simplify_tolerance: Tolerance for contour simplification.
            Higher values produce simpler polygons with fewer points.

    Returns:
        Dict mapping label -> list of contour arrays.
        Each contour is shape (N, 2) with (x, y) coordinates.
    """
    labels = np.unique(mask)
    labels = labels[labels != 0]

    contours = {}

    for label in labels:
        cell_mask = (mask == label).astype(np.uint8)

        cell_contours = measure.find_contours(cell_mask, level=0.5)

        if not cell_contours:
            continue

        simplified = []
        for contour in cell_contours:
            # find_contours returns (row, col) = (y, x), swap to (x, y)
            contour_xy = contour[:, ::-1]

            if simplify_tolerance > 0:
                contour_xy = _simplify_contour(contour_xy, simplify_tolerance)

            if len(contour_xy) >= 3:
                simplified.append(contour_xy)

        if simplified:
            contours[int(label)] = simplified

    return contours


def _simplify_contour(contour, tolerance):
    """Simplify a contour using Douglas-Peucker algorithm."""
    if len(contour) <= 3:
        return contour

    try:
        from skimage.measure import approximate_polygon
        return approximate_polygon(contour, tolerance=tolerance)
    except ImportError:
        step = max(1, int(tolerance))
        return contour[::step]


def clean_and_close_contours(contours):
    """Ensure contours are closed for GeoJSON compatibility."""
    cleaned = {}

    for label, label_contours in contours.items():
        closed_contours = []
        for contour in label_contours:
            if len(contour) < 3:
                continue

            if not np.allclose(contour[0], contour[-1]):
                contour = np.vstack([contour, contour[0:1]])

            closed_contours.append(contour)

        if closed_contours:
            cleaned[label] = closed_contours

    return cleaned


def _contour_to_geojson_coords(contour):
    """Convert contour array to GeoJSON coordinate list."""
    return [[float(x), float(y)] for x, y in contour]


def _mask_features(mask, object_type, simplify_tolerance):
    contours = clean_and_close_contours(mask_to_contours(mask, simplify_tolerance))
    features = []
    for label, label_contours in contours.items():
        if len(label_contours) == 1:
            coords = [_contour_to_geojson_coords(label_contours[0])]
        else:
            # multiple contours: largest is exterior, rest are holes
            areas = [len(c) for c in label_contours]
            main_idx = int(np.argmax(areas))
            exterior = _contour_to_geojson_coords(label_contours[main_idx])
            holes = [_contour_to_geojson_coords(c)
                     for i, c in enumerate(label_contours) if i != main_idx]
            coords = [exterior] + holes
        features.append({
            "type": "Feature",
            "id": f"{object_type}_{label}",
            "geometry": {"type": "Polygon", "coordinates": coords},
            "properties": {"objectType": object_type, "measurements": {"label": int(label)}},
        })
    return features


def build_frame_geojson(cell2d, art2d, simplify_tolerance=1.0):
    """Full-fidelity per-frame record: cells and artifacts as separate, independently-collapsed,
    potentially overlapping polygon features. Nothing here is excluded or occluded -- that only
    happens in the flat training mask (scene.build_training_mask)."""
    features = (_mask_features(cell2d, "cell", simplify_tolerance)
                + _mask_features(art2d, "artifact", simplify_tolerance))
    return {"type": "FeatureCollection", "features": features}


def build_prediction_geojson(cell_lab, nuc_lab, props, type_names=None, context_names=None,
                             scale=1.0, offset_xy=(0.0, 0.0), simplify_tolerance=1.0):
    """Predicted instances from the query head (model_v3.decode_queries) as QuPath CELL objects:
    one feature per cell with its body as `geometry`, its nucleus as `nucleusGeometry` (the same
    query owns both, so they cannot be mismatched), and the cell type + tissue context the SAME query
    predicted as `classification` / `measurements`. Coordinates are model pixels mapped by
    `scale` and `offset_xy` (x, y) -- e.g. scale = 0.325 / 0.17 and the crop's top-left corner put the
    polygons back onto the full-resolution slide. A type is a slot index unless `type_names` names
    the slots; slot identities are per tile (they are matched, not fixed classes), so compare types
    across tiles by their predicted expression, not by the slot number."""
    def _poly(mask):
        cs = clean_and_close_contours(mask_to_contours(mask.astype(np.int32), simplify_tolerance))
        if 1 not in cs:
            return None
        rings = sorted(cs[1], key=len, reverse=True)
        tr = lambda c: [[float(x) * scale + offset_xy[0], float(y) * scale + offset_xy[1]] for x, y in c]
        return {"type": "Polygon", "coordinates": [tr(r) for r in rings]}

    features = []
    for i in range(1, int(cell_lab.max()) + 1):
        geom = _poly(cell_lab == i)
        if geom is None:
            continue
        t, c = int(props["type"][i - 1]), int(props["context"][i - 1])
        tname = type_names[t] if type_names is not None else f"T{t:02d}"
        feat = {
            "type": "Feature",
            "id": f"cell_{i}",
            "geometry": geom,
            "properties": {
                "objectType": "cell",
                "classification": {"name": tname},
                "measurements": {
                    "label": i,
                    "existence": float(props["score"][i - 1]),
                    "type_prob": float(props["type_prob"][i - 1]),
                    "context_idx": c,
                },
                "context": context_names[c] if context_names is not None else c,
            },
        }
        ngeom = _poly(nuc_lab == i)
        if ngeom is not None:
            feat["nucleusGeometry"] = ngeom
        features.append(feat)
    return {"type": "FeatureCollection", "features": features}
