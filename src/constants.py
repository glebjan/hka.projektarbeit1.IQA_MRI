from pathlib import Path

RESNET50 = Path("models/RadImageNet_pytorch/ResNet50.pt")

# DreamSim downloads its ViT/LoRA weights here on first use (~1 GB for the
# default "ensemble" type). Excluded from git via .gitignore — unlike
# RESNET50, these weights are fetched automatically and need not be tracked.
DREAMSIM_CACHE = Path("models/dreamsim")

REPORT = Path("report") / "IXI661_report.csv"
