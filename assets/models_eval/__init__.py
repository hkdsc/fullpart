import json
import os
from .iq_aigc_model import *
from .vq_aigc_model import *
from .umt_score_model import *
from .motionsmooth_model import *


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dim_mean_std.json')) as json_file:
    dim_mean_std = json.load(json_file)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'overall_coeff.json')) as json_file:
    overall_coeff = json.load(json_file)

def get_overall_score(iq_score, vq_score, umt_score, motionsmooth_score):
    iq_score = (iq_score - dim_mean_std["iq_aigc"][2]) / (dim_mean_std["iq_aigc"][3] - dim_mean_std["iq_aigc"][2])
    vq_score = (vq_score - dim_mean_std["vq_aigc"][2]) / (dim_mean_std["vq_aigc"][3] - dim_mean_std["vq_aigc"][2])
    umt_score = (umt_score - dim_mean_std["umt_score"][2]) / (dim_mean_std["umt_score"][3] - dim_mean_std["umt_score"][2])
    motionsmooth_score = (motionsmooth_score - dim_mean_std["motion_smoothness"][2]) / (dim_mean_std["motion_smoothness"][3] - dim_mean_std["motion_smoothness"][2])

    overall_score = overall_coeff["bias"]
    overall_score += overall_coeff["iq_aigc"] * iq_score
    overall_score += overall_coeff["vq_aigc"] * vq_score
    overall_score += overall_coeff["umt_score"] * umt_score
    overall_score += overall_coeff["motion_smoothness"] * motionsmooth_score
    return overall_score
