# Cooling Tower Synthetic Data Generator Project

### CNN Model TODO:
- create eval process script for testing (fouling counter, eval process, new timeline column, etc.)
- put synth dataset on server and make new list.txt for it.
- ensure deterministic seed is working!
- add more torchmetrics: 
- https://lightning.ai/docs/torchmetrics/stable/classification/precision_recall_curve.html
- https://lightning.ai/docs/torchmetrics/stable/classification/precision_recall_curve.html#torchmetrics.classification.BinaryPrecisionRecallCurve
- https://lightning.ai/docs/torchmetrics/stable/segmentation/mean_iou.html
- https://lightning.ai/docs/torchmetrics/stable/segmentation/generalized_dice.html

### Synthetic Data Generator TODO:
- ensure all synthetic data generation is deterministic using seeds!
- find new way to algorithmically grow dust spots (more linear).
- (maybe) depending on above algorithm, incorporate script C into script A.
- (when there is time) restructure and reorganize all utils for this directory.

### Academic Paper TODO:
- Touch up method_diagram so that arrows don't directly cross (use jump).