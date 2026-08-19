import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


def _load_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class LEFuseTeacher(nn.Module):
    """Training-only frozen LEFuse teacher.

    The wrapper loads only LEFuse's Unet_fuser forward model. It deliberately
    avoids constructing LEFuse's VGG perceptual-loss member, so VGG is never
    part of the teacher object or the LUT-Fuse inference graph.
    """

    output_space = "rgb"
    output_range = (0.0, 1.0)

    def __init__(self, source_dir, checkpoint, device):
        super().__init__()
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()

        inference_path = self.source_dir / "inference.py"
        if not inference_path.is_file():
            raise FileNotFoundError(
                f"LEFuse inference.py was not found: {inference_path}"
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"LEFuse checkpoint was not found: {self.checkpoint_path}"
            )

        source_dir_string = str(self.source_dir)
        path_was_added = source_dir_string not in sys.path
        if path_was_added:
            sys.path.insert(0, source_dir_string)
        try:
            self._runtime = _load_module(
                inference_path,
                "_lut_fuse_lefuse_inference_runtime",
            )
        finally:
            if path_was_added:
                sys.path.remove(source_dir_string)

        net_module_name = self._runtime.LEFuse.__module__
        net_module = sys.modules.get(net_module_name)
        if net_module is None or not hasattr(net_module, "Unet_fuser"):
            raise ImportError(
                "Could not resolve Unet_fuser from the LEFuse source tree."
            )

        self.fuser = net_module.Unet_fuser(dim=32).to(device)
        self._load_fuser_weights(device)

        for parameter in self.fuser.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def _load_fuser_weights(self, device):
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        if not isinstance(state_dict, dict):
            raise TypeError("LEFuse checkpoint must contain a state dict.")

        expected_keys = set(self.fuser.state_dict().keys())
        fuser_state = {}
        for original_key, value in state_dict.items():
            key = original_key
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("fuser."):
                key = key[len("fuser."):]
            if key in expected_keys:
                fuser_state[key] = value

        if not fuser_state:
            raise KeyError(
                "No Unet_fuser weights were found in the LEFuse checkpoint."
            )

        missing, unexpected = self.fuser.load_state_dict(
            fuser_state,
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(
                "LEFuse fuser checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def train(self, mode=True):
        # The teacher must remain in eval mode even when the student trainer
        # switches its own modules to train mode.
        super().train(False)
        self.fuser.eval()
        return self

    @staticmethod
    def _validate_inputs(visible, infrared):
        if visible.ndim != 4 or visible.shape[1] != 3:
            raise ValueError(
                f"Expected visible [B,3,H,W], got {tuple(visible.shape)}"
            )
        if infrared.ndim != 4 or infrared.shape[1] not in (1, 3):
            raise ValueError(
                f"Expected infrared [B,1,H,W] or [B,3,H,W], "
                f"got {tuple(infrared.shape)}"
            )
        if visible.shape[0] != infrared.shape[0]:
            raise ValueError("Visible/infrared batch sizes do not match.")
        if visible.shape[-2:] != infrared.shape[-2:]:
            raise ValueError("Visible/infrared spatial sizes do not match.")
        if visible.device != infrared.device:
            raise ValueError("Visible/infrared tensors must share a device.")

        tolerance = 1e-4
        for name, tensor in (("visible", visible), ("infrared", infrared)):
            minimum = float(tensor.detach().amin().cpu())
            maximum = float(tensor.detach().amax().cpu())
            if minimum < -tolerance or maximum > 1.0 + tolerance:
                raise ValueError(
                    f"{name} must be in [0,1], got min={minimum}, max={maximum}"
                )

    @torch.no_grad()
    def forward(self, visible, infrared):
        self._validate_inputs(visible, infrared)
        outputs = []

        # ETRI's fixed nighttime preprocessing currently accepts one image at
        # a time because OpenCV NLM is CPU based. This is training-only code.
        for index in range(visible.shape[0]):
            visible_sample = visible[index:index + 1]
            infrared_sample = infrared[index:index + 1]

            visible_y, cr, cb = self._runtime.RGB2YCrCb(visible_sample)
            if infrared_sample.shape[1] == 1:
                infrared_y = infrared_sample
            else:
                infrared_y, _, _ = self._runtime.RGB2YCrCb(infrared_sample)

            visible_y_processed = (
                self._runtime.dark_region_edge_aware_nlm_denoise_y(
                    vi_y=visible_y,
                    ir_y=infrared_y,
                )
            )
            cr_processed, cb_processed = (
                self._runtime.dark_region_chroma_process(
                    cr=cr,
                    cb=cb,
                    vi_y=visible_y,
                )
            )

            fused_y = self.fuser(visible_y_processed, infrared_y)
            fused_y = self._runtime.normalize_fused_y(
                fused_y,
                normalize_mode=self._runtime.NORMALIZE_MODE,
            )
            fused_rgb = self._runtime.YCbCr2RGB(
                fused_y,
                cb_processed,
                cr_processed,
            )
            outputs.append(torch.clamp(fused_rgb, 0.0, 1.0))

        return torch.cat(outputs, dim=0)
