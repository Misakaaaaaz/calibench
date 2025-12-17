from Uncalibrated import Uncalibrated
from TS import TS
from PTS import PTS
from CTS import CTS
from ETS import ETS
from SMART_ import SMART_
from HB import HB
from BBQ import BBQ
from VS import VS
from GC import GC
from ProCal_DR import ProCal_DR
from FC import FC
from Spline import Spline

def evaluate(pt, seed, dataset, model, vs, loss):
    # Uncalibrated(pt)
    # TS(pt)
    PTS(pt, seed)
    # CTS(pt)
    # ETS(pt)
    # SMART_(pt, dataset, model, seed, vs, loss)
    # HB(pt)
    # BBQ(pt)
    # VS(pt)
    # GC(pt)
    # ProCal_DR(pt)
    # FC(pt, model, dataset)
    # Spline(pt)

if __name__ == '__main__':
    # pt = 'output/imagenet_sketch_resnet50_seed4_vs0.2.pt'
    # seed = 4
    # dataset = 'imagenet_sketch'
    # model = 'resnet50'
    # vs = 0.2
    # loss = 'ce'
    # evaluate(pt, seed, dataset, model, vs, loss)
    evaluate(pt='output/imagenet_sketch_swin_b_seed5_vs0.2.pt', seed=5, dataset='imagenet_sketch', model='swin_b', vs=0.2, loss='ce')