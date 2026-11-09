# Backend-independent conv-weight layout helpers, shared by the keras and torch
# pattern utilities. Pure Python: the actual tensor transpose stays in the backends.

from pquant.core.constants import CANONICAL_CONV_LAYOUT, CONV_LAYOUT_AXES


def layout_to_axes(layout):
    if len(layout) != 4 or set(layout) != set("HWIO"):
        raise ValueError(f"layout must be a permutation of 'HWIO', got {layout!r}")
    return tuple(CONV_LAYOUT_AXES[ch] for ch in layout)


def layout_perm(src, dst=CANONICAL_CONV_LAYOUT):
    """Permutation tuple that reorders axes from `src` layout to `dst` layout."""
    s = layout_to_axes(src)
    d = layout_to_axes(dst)
    return tuple(s.index(ax) for ax in d)
