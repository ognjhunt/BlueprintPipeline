"""
DWM bundle packaging utilities.

Packages all DWM conditioning inputs (static scene video, hand mesh video,
text prompts) into a structured bundle ready for DWM inference.
"""

from .dwm_bundle import (
    DWMBundlePackager,
    create_bundle_manifest,
    package_dwm_bundle,
)

__all__ = [
    "DWMBundlePackager",
    "create_bundle_manifest",
    "package_dwm_bundle",
]
