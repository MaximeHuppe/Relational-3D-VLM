"""Stage A and Stage B networks.

Stage A (``shape_segmenter``) segments any requested shape name from the binary
scene volume. Stage B (``relational_vlm`` and the modules it composes) segments
the target described by three relations to three named anchors, and never sees
the scene, the labels or the target. ``anchor_provider`` is where the two meet:
it decides whether Stage B's three channels are ground truth or Stage A's
predictions.
"""
