"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("stacked-barretts-esophagus-endoscopy-images",): interface_0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interface_0_handler():
    # Read the input
    input_stacked_barretts_esophagus_endoscopy_images = load_image_file_as_array(
        location=INPUT_PATH / "images/stacked-barretts-esophagus-endoscopy",
    )
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    """ Run your model here (MINIMAL PATCH) """
    # 1) ckpt 경로: resources/ → 없으면 /opt/ml/models/ 에서 fallback
    import os
    from glob import glob
    from pathlib import Path
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision.models import resnet50
    import numpy as np

    def _find_ckpt():
        # 우선순위: env CKPT_PATH -> resources/CKPT_NAME -> resources/*.pth -> /opt/ml/models/*.pth
        env_p = os.getenv("CKPT_PATH", "")
        if env_p and Path(env_p).exists():
            return Path(env_p)
        ckpt_name = os.getenv("CKPT_NAME", "run1_1.pth")
        for base in [RESOURCE_PATH, Path("/opt/ml/models")]:
            p = base / ckpt_name
            if p.exists():
                return p
            cands = sorted(list(base.glob("*.pth"))) + sorted(list(base.glob("*.pt")))
            if cands:
                return cands[0]
        raise FileNotFoundError("No checkpoint found in resources/ or /opt/ml/models/. Set CKPT_PATH or include a .pth")

    ckpt_path = _find_ckpt()
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    # 2) 모델 생성(베이스라인과 동일 흐름 유지: 그냥 여기서 직접 돌림)
    net = resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 1)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("ema_state_dict") or ckpt.get("state_dict")
    if state_dict is None:
        raise RuntimeError("Checkpoint missing 'state_dict'/'ema_state_dict'")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    print(f"[load] missing={list(missing)} unexpected={list(unexpected)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()

    # 3) 로짓→확률 후처리(학습 시 저장된 보정/LA 정보 반영)
    la_bias = float(ckpt.get("la_logit_bias", 0.0))
    args_dict = ckpt.get("args", {}) or {}
    la_used = bool(args_dict.get("la", False))
    calib = ckpt.get("calib", {"type": "none"}) or {"type": "none"}
    ddts_state = ckpt.get("ddts_state", None)

    import torch.nn.functional as F

    class _DDTS(nn.Module):
        def __init__(self, hidden=8):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear( hidden, 1))
        def forward(self, z):
            z = z.view(-1,1)
            T = 1.0 + F.softplus(self.mlp(z))
            return z / T

    def _postproc(logits: torch.Tensor) -> torch.Tensor:
        z = logits.view(-1,1)
        # DDTS 경로
        if calib.get("type") == "ddts":
            if ddts_state is None:
                raise RuntimeError("calib.type == 'ddts' but ddts_state missing")
            ddts = _DDTS(hidden=int(ddts_state.get("hidden", 8)))
            ddts.load_state_dict(ddts_state["state_dict"])
            ddts.to(device).eval()
            if bool(ddts_state.get("la_applied", False)) and la_used and (la_bias != 0.0):
                z = z + float(la_bias)
            z_cal = ddts(z)
            return torch.sigmoid(z_cal).view(-1).cpu()

        # Non-DDTS 경로: p = sigmoid(z + LA), 이후 보정
        if la_used and (la_bias != 0.0):
            z = z + float(la_bias)
        base_p = torch.sigmoid(z).view(-1).cpu().numpy()

        def _sigmoid_np(x): return 1.0/(1.0+np.exp(-x))
        def _logit_np(p):
            p = np.clip(p, 1e-6, 1-1e-6)
            return np.log(p/(1-p))

        t = calib.get("type","none")
        if t == "platt":
            A, B = float(calib["A"]), float(calib["B"])
            zc = _logit_np(base_p)*A + B
            p = _sigmoid_np(zc)
            return torch.from_numpy(p)
        elif t == "beta":
            a, b, g = float(calib["alpha"]), float(calib["beta"]), float(calib["gamma"])
            zc = a*np.log(np.clip(base_p,1e-6,1-1e-6)) + b*np.log(np.clip(1-base_p,1e-6,1-1e-6)) + g
            p = _sigmoid_np(zc)
            return torch.from_numpy(p)
        elif t == "ccc":
            A1,B1,A0,B0 = float(calib["A1"]),float(calib["B1"]),float(calib["A0"]),float(calib["B0"])
            z = _logit_np(base_p)
            p1p = _sigmoid_np(A1*z + B1)
            p0p = _sigmoid_np(A0*(-z) + B0)
            denom = np.clip(p1p+p0p, 1e-6, None)
            return torch.from_numpy(p1p/denom)
        else:
            return torch.from_numpy(base_p)

    # 4) 전처리(베이스라인 흐름 유지)
    tfm = T.Compose([
        T.ToPILImage(),
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    arr = input_stacked_barretts_esophagus_endoscopy_images
    # 스택 표준화: (Z,H,W) or (Z,H,W,C)로 간주. 2D면 Z=1
    if arr.ndim == 2:  # (H,W)
        arr = arr[None, ...]
    if arr.ndim == 3:
        # (Z,H,W)로 가정. 만약 (H,W,C)였다면 아래 변환에서 자동 대응
        pass
    # 예측
    probs = []
    with torch.no_grad():
        for i in range(arr.shape[0]):
            frame = arr[i]
            # (H,W) or (H,W,C) → HWC3
            if frame.ndim == 2:
                frame = np.stack([frame,frame,frame], axis=-1)
            elif frame.ndim == 3:
                if frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                elif frame.shape[-1] > 3:
                    frame = frame[...,:3]
            # dtype 정규화
            if frame.dtype != np.uint8:
                fmin, fmax = float(np.min(frame)), float(np.max(frame))
                if fmax > fmin:
                    frame = ((frame - fmin)/(fmax - fmin) * 255.0).astype(np.uint8)
                else:
                    frame = np.zeros_like(frame, dtype=np.uint8)
            x = tfm(frame).unsqueeze(0).to(device)
            logits = net(x).view(-1)
            p = _postproc(logits)[0].item()
            # 안전 보정
            if not np.isfinite(p): p = 0.0
            p = float(min(1.0, max(0.0, p)))
            probs.append(p)

    # 베이스라인과 동일한 출력 형태 유지
    output_stacked_neoplastic_lesion_likelihoods = [float(p) for p in probs]

    write_json_file(
        location=OUTPUT_PATH / "stacked-neoplastic-lesion-likelihoods.json",
        content=output_stacked_neoplastic_lesion_likelihoods,
    )

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

0
def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
