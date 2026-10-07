"""Portable release locations, supplied by code/run.py."""
import os
from pathlib import Path
REPO = Path(os.environ['BREASTDM_ROOT']).resolve()
RESULT = REPO / 'result'
RUNS = Path(os.environ.get('BREASTDM_RUNS', str(RESULT))).resolve()
OUTPUT = Path(os.environ.get('BREASTDM_OUTPUT', str(RESULT / 'reproduced'))).resolve()
PAPER = RESULT / 'paper_outputs'
PRETRAINED = REPO / 'code' / 'pretrained'
EXP1_DATA = REPO / 'data/classification_exp1/processed_dataset'
EXP2_DATA = REPO / 'data/classification_exp2/img17Se'
SEG2_DATA = REPO / 'data/segmentation_2d/seg2D_clean_v2'
SEG3_DATA = REPO / 'data/segmentation_3d/seg3D_clean'
RUN3 = RESULT / 'seg3d_unet_locked_volume_ppv_seed42_20260911'
RUNTIME = Path(__file__).resolve().parent
