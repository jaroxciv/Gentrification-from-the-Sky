"""Models — Deep-learning change detection.

Detects physical urban change between the two annual composites. Compares the
six methodologies from the paper (§3.3): Simple-Diff, Res-Net, FC-SiamDiff,
CGNet, Bi-Temporal Siamese (BiDateNet), and TinyCD. Each model produces a
per-band change map; Otsu thresholding turns features into binary change.
"""
