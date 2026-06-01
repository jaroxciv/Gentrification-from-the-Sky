"""X — Satellite feature preparation.

Builds cloud-masked Sentinel-2 annual median composites over Greater London for
the two study years via Google Earth Engine, then clips/merges/tiles them into
the 256x256 imagelets consumed by the change-detection models (paper §3.2).
"""
