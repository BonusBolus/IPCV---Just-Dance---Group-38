# PyInstaller recipe: makes the folder dist/JustDance with the program JustDance (JustDance.exe on Windows).
#
# Build (see README, "Make an executable"):
#     python -m PyInstaller just_dance.spec --noconfirm
#
# Use a Python environment with the CPU version of PyTorch: the CUDA version adds about 4 GB of
# NVIDIA libraries that the game does not use (it runs YOLO with OpenVINO on the CPU).
# PyInstaller makes a program for the system it runs on: build on Windows for Windows.
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH)
MODEL = ROOT / "yolo26n-pose.pt"
OPENVINO_MODEL = ROOT / "yolo26n-pose_openvino_model"

if not OPENVINO_MODEL.exists():
    # the game loads this folder (pose_processing.load_pose_model); make it once before packing
    from ultralytics import YOLO
    YOLO(str(MODEL)).export(format="openvino", dynamic=True)

# files the game reads, at the same place as in the project (config.ROOT points to the package folder)
datas = [
    (str(MODEL), "."),
    (str(OPENVINO_MODEL), OPENVINO_MODEL.name),
    (str(ROOT / "face_tracking" / "face_detection_yunet_2023mar.onnx"), "face_tracking"),
]
datas += [(str(path), "choreography") for path in sorted((ROOT / "choreography").glob("*.json"))]
datas += [(str(path), "choreography") for path in sorted((ROOT / "choreography").glob("*.ogg"))]

# Ultralytics, OpenVINO and torchvision load parts of themselves at runtime (settings files, CPU
# plugins, torchvision's compiled operators: new versions call them _C_stable.so, which PyInstaller's
# own torchvision rules do not pack yet -> "operator torchvision::nms does not exist")
binaries, hiddenimports = [], []
for package in ("ultralytics", "openvino", "torchvision"):
    package_datas, package_binaries, package_imports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_imports

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "torch.utils.tensorboard", "tensorboard", "IPython", "jupyter", "notebook"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="JustDance", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="JustDance")
