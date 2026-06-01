"""Change-detection model architectures, one module per method.

Each module exposes a model class (and where relevant a feature-extraction
entry point) for one of the methodologies compared in the paper. They share the
training loop, patching and thresholding utilities in ``gfs.change_detection``.
"""
