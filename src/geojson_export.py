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
