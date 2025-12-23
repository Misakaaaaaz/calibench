from types import SimpleNamespace

import torch

from bench.main import generate_cali_result


def test_bench():
    args = SimpleNamespace()
    RUN_METHODS = ["uncalibrated", "TS", "PTS", "CTS", "ETS", "SMART", "HB", "BBQ", "VS", "GC", "ProCal_DR", "FC",
                   "Spline"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_cali_result(args,'imagenet_lt',"D:\storage\Confidence Calibration\datasets\ImageNet-LT","resnet50",1,0.2,0.2,device,RUN_METHODS)

if __name__ == "__main__":
    test_bench()