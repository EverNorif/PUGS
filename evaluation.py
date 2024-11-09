import json
import numpy as np

from settings import get_args


# preds should be Nx2, gts should be N
def ADE(preds, gts):
    point_preds = np.mean(preds, axis=1)
    ade = np.abs(point_preds - gts)
    return np.mean(ade)

def ALDE(preds, gts):
    point_preds = np.mean(preds, axis=1)
    alde = np.abs(np.log(point_preds) - np.log(gts))
    return np.mean(alde)

def APE(preds, gts):
    point_preds = np.mean(preds, axis=1)
    ape = np.abs(point_preds - gts) / gts
    return np.mean(ape)

def MnRE(preds, gts):
    point_preds = np.mean(preds, axis=1)
    p_over_t = point_preds / gts
    t_over_p = gts / point_preds
    ratios = np.vstack([p_over_t, t_over_p])
    mnre = np.min(ratios, axis=0)
    return np.mean(mnre)
    

def show_metrics(preds, gts):
    print('ADE  %.3f' % ADE(preds, gts))
    print('ALDE %.3f' % ALDE(preds, gts))
    print('APE  %.3f' % APE(preds, gts))
    print('MnRE %.3f' % MnRE(preds, gts))


if __name__ == '__main__':
    
    args = get_args()

    with open(args.preds_json_path, 'r') as f:
        preds_dict = json.load(f)
    with open(args.gts_json_path, 'r') as f:
        gts_dict = json.load(f)
    
    preds = np.zeros((len(preds_dict), 2))
    gts = np.zeros(len(preds_dict))
    for i, (k, v) in enumerate(preds_dict.items()):
        preds[i] = [v[0], v[1]]
        gts[i] = gts_dict[k.split('_')[0]]

    show_metrics(preds, gts)
