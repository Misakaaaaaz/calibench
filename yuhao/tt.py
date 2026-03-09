import timm
from timm.models._registry import list_pretrained, get_pretrained_cfg

arch = "eva02_large_patch14_224"
print(list_pretrained(arch))   # 会打印出所有可用的 “arch.tag” 字符串