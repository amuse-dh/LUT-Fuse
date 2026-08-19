import argparse
import datetime as dt
import json
import os
import random
import subprocess
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import transforms as T
from data.o_fusion_dataset import DistillDataSet, RandomCropPair
from fine_tune_lut import TV_4D, save_generator_context, save_lut
from scripts.calculate import (
    Generator_for_info,
    OptimizableLUT,
    apply_fusion_4d_with_interpolation,
)
from scripts.loss_lut import fusion_loss
from teachers import LEFuseTeacher


ALLOWED_TEACHERS = {"original_mmnet", "lefuse"}
ALLOWED_LOSSES = {"dist_int", "dist_ssim", "lut_tv", "lut_monotonic"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="A0/A1 distillation trainer for LUT-Fuse."
    )
    parser.add_argument("--config", required=True, help="YAML experiment config")
    return parser.parse_args()


def project_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The config root must be a mapping.")
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_weights(config):
    configured = config.get("loss", {})
    unknown_enabled = [
        name
        for name, options in configured.items()
        if name not in ALLOWED_LOSSES
        and isinstance(options, dict)
        and options.get("enabled", False)
    ]
    if unknown_enabled:
        raise ValueError(
            "A1 only supports the original distillation/regularization losses; "
            f"disable unsupported losses: {unknown_enabled}"
        )

    result = {}
    for name in ALLOWED_LOSSES:
        options = configured.get(name, {})
        result[name] = (
            float(options.get("weight", 0.0))
            if options.get("enabled", False)
            else 0.0
        )
    if not any(result.values()):
        raise ValueError("At least one supported loss must have a non-zero weight.")
    return result


def file_stems(paths):
    return [Path(path).stem for path in paths]


def validate_dataset_contract(dataset, split, needs_cached_teacher, match_stems):
    visible_count = len(dataset.visible_files)
    infrared_count = len(dataset.infrared_files)
    teacher_count = len(dataset.other_fuse_files)

    if visible_count == 0 or infrared_count == 0:
        raise ValueError(f"{split}: visible and infrared folders must not be empty.")
    if visible_count != infrared_count:
        raise ValueError(
            f"{split}: visible/infrared counts differ: "
            f"{visible_count} vs {infrared_count}"
        )
    if needs_cached_teacher and teacher_count != visible_count:
        raise ValueError(
            f"{split}: cached teacher count must equal input count: "
            f"{teacher_count} vs {visible_count}"
        )

    if match_stems:
        visible_stems = file_stems(dataset.visible_files)
        infrared_stems = file_stems(dataset.infrared_files)
        if visible_stems != infrared_stems:
            raise ValueError(f"{split}: visible/infrared sample IDs are not aligned.")
        if needs_cached_teacher and visible_stems != file_stems(
            dataset.other_fuse_files
        ):
            raise ValueError(f"{split}: cached teacher sample IDs are not aligned.")


def build_dataset(config, split, teacher_type):
    split_config = config["dataset"][split]
    teacher_dir = split_config.get("teacher_dir")
    needs_cached_teacher = teacher_type == "original_mmnet"
    if needs_cached_teacher and not teacher_dir:
        raise ValueError(
            f"dataset.{split}.teacher_dir is required for original_mmnet."
        )

    if split == "train":
        crop_size = tuple(config["training"].get("crop_size", [128, 128]))
        transform = RandomCropPair(size=crop_size)
    else:
        transform = T.Compose([T.Resize_16(), T.ToTensor()])

    dataset = DistillDataSet(
        visible_path=str(project_path(split_config["visible_dir"])),
        infrared_path=str(project_path(split_config["infrared_dir"])),
        other_fuse_path=(
            str(project_path(teacher_dir)) if needs_cached_teacher else None
        ),
        phase=split,
        transform=transform,
    )
    validate_dataset_contract(
        dataset,
        split,
        needs_cached_teacher,
        config["dataset"].get("require_matching_stems", True),
    )
    return dataset


def build_loaders(config, teacher_type, device):
    training = config["training"]
    batch_size = int(training["batch_size"])
    num_workers = int(training.get("num_workers", 0))
    train_dataset = build_dataset(config, "train", teacher_type)
    val_dataset = build_dataset(config, "val", teacher_type)
    common = {
        "pin_memory": device.type == "cuda",
        "num_workers": num_workers,
    }
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            **common,
        ),
        DataLoader(
            val_dataset,
            batch_size=int(training.get("val_batch_size", 1)),
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            **common,
        ),
    )


def load_student(config, device):
    lut_path = project_path(config["student"]["lut_checkpoint"])
    context_path = project_path(config["student"]["context_checkpoint"])
    if not lut_path.is_file() or not context_path.is_file():
        raise FileNotFoundError(
            f"Student checkpoint missing: LUT={lut_path}, context={context_path}"
        )

    lut_tensor = torch.from_numpy(np.load(lut_path).astype(np.float32)).to(device)
    lut_model = OptimizableLUT(lut_tensor).to(device)
    context = Generator_for_info().to(device)
    context.load_state_dict(torch.load(context_path, map_location=device))
    return lut_model, context


def build_teacher(config, device):
    teacher_config = config["teacher"]
    teacher_type = teacher_config["type"]
    if teacher_type not in ALLOWED_TEACHERS:
        raise ValueError(
            f"teacher.type must be one of {sorted(ALLOWED_TEACHERS)}, "
            f"got {teacher_type!r}"
        )
    if teacher_type == "original_mmnet":
        return teacher_type, None
    if not teacher_config.get("freeze", True):
        raise ValueError("A1 requires teacher.freeze: true.")

    teacher = LEFuseTeacher(
        source_dir=project_path(teacher_config["source_dir"]),
        checkpoint=project_path(teacher_config["checkpoint"]),
        device=device,
    ).to(device)
    teacher.eval()
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("LEFuse teacher parameters must all be frozen.")
    return teacher_type, teacher


def get_teacher_target(teacher, cached_target, visible, infrared, device):
    if teacher is None:
        if cached_target is None:
            raise ValueError("The original_mmnet run requires cached teacher images.")
        return cached_target.to(device, non_blocking=True)
    teacher.eval()
    with torch.no_grad():
        return teacher(visible, infrared).detach()


def tensor_minmax(tensor):
    detached = tensor.detach()
    return float(detached.amin().cpu()), float(detached.amax().cpu())


def assert_range(name, tensor, minimum, maximum, tolerance=1e-4):
    actual_min, actual_max = tensor_minmax(tensor)
    assert_finite(name, tensor)
    if actual_min < minimum - tolerance or actual_max > maximum + tolerance:
        raise ValueError(
            f"{name} expected in [{minimum},{maximum}], "
            f"got [{actual_min:.6f},{actual_max:.6f}]"
        )


def assert_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def check_and_print_contract(visible, infrared, teacher_target, student_output):
    if visible.ndim != 4 or visible.shape[1] != 3:
        raise ValueError(f"visible must be [B,3,H,W], got {tuple(visible.shape)}")
    if infrared.ndim != 4 or infrared.shape[1] != 1:
        raise ValueError(f"infrared must be [B,1,H,W], got {tuple(infrared.shape)}")
    if teacher_target.shape != student_output.shape:
        raise ValueError(
            "Teacher/student shapes differ: "
            f"{tuple(teacher_target.shape)} vs {tuple(student_output.shape)}"
        )
    if teacher_target.shape != visible.shape:
        raise ValueError(
            "Teacher/student outputs must be RGB and spatially aligned to visible."
        )

    assert_range("visible", visible, 0.0, 1.0)
    assert_range("infrared", infrared, 0.0, 1.0)
    assert_range("teacher_target", teacher_target, 0.0, 1.0)
    # The upstream checkpoint's LUT itself is not range constrained, so the
    # reconstructed student RGB can leave [0,1]. Preserve that behavior and
    # reject only non-finite output instead of silently clamping it.
    assert_finite("student_output", student_output)

    print("[Tensor contract: first training batch]")
    for name, tensor in (
        ("visible RGB", visible),
        ("infrared Y", infrared),
        ("teacher RGB", teacher_target),
        ("student RGB (unclamped)", student_output),
    ):
        minimum, maximum = tensor_minmax(tensor)
        print(
            f"  {name}: shape={tuple(tensor.shape)}, "
            f"range=[{minimum:.6f}, {maximum:.6f}]"
        )


def compute_loss_terms(
    visible,
    infrared,
    teacher_target,
    student_output,
    lut,
    structural_loss,
    regularizer,
    weights,
):
    zero = student_output.new_zeros(())
    tv, monotonic = regularizer(lut)
    if student_output.shape != teacher_target.shape:
        raise ValueError(
            "Teacher/student shapes differ: "
            f"{tuple(teacher_target.shape)} vs {tuple(student_output.shape)}"
        )
    terms = {
        "dist_int": (
            F.l1_loss(student_output, teacher_target)
            if weights["dist_int"]
            else zero
        ),
        # This is the repository's original visible/IR-to-student structural
        # fusion loss. It is retained exactly; it is not teacher/student SSIM.
        "dist_ssim": (
            structural_loss(visible, infrared, student_output)
            if weights["dist_ssim"]
            else zero
        ),
        "lut_tv": tv if weights["lut_tv"] else zero,
        "lut_monotonic": monotonic if weights["lut_monotonic"] else zero,
    }
    total = sum(weights[name] * value for name, value in terms.items())
    return total, terms


def accumulate(sums, total, terms, batch_size):
    sums["total"] += float(total.detach().cpu()) * batch_size
    for name, value in terms.items():
        sums[name] += float(value.detach().cpu()) * batch_size


def averages(sums, sample_count):
    return {name: value / sample_count for name, value in sums.items()}


def run_epoch(
    loader,
    lut_model,
    context,
    teacher,
    device,
    structural_loss,
    regularizer,
    weights,
    optimizer=None,
    debug_contract=False,
):
    is_training = optimizer is not None
    lut_model.train(is_training)
    context.train(is_training)
    if teacher is not None:
        teacher.eval()

    sums = {"total": 0.0, **{name: 0.0 for name in ALLOWED_LOSSES}}
    sample_count = 0
    contract_pending = debug_contract
    grad_context = torch.enable_grad() if is_training else torch.no_grad()

    with grad_context:
        for visible, infrared, cached_target, _ in loader:
            visible = visible.to(device, non_blocking=True)
            infrared = infrared.to(device, non_blocking=True)
            teacher_target = get_teacher_target(
                teacher, cached_target, visible, infrared, device
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            lut = lut_model()
            output = apply_fusion_4d_with_interpolation(
                visible * 255.0,
                infrared * 255.0,
                lut,
                context,
            )
            if contract_pending:
                check_and_print_contract(visible, infrared, teacher_target, output)
                contract_pending = False

            total, terms = compute_loss_terms(
                visible,
                infrared,
                teacher_target,
                output,
                lut,
                structural_loss,
                regularizer,
                weights,
            )
            if is_training:
                total.backward()
                optimizer.step()

            batch_size = visible.shape[0]
            accumulate(sums, total, terms, batch_size)
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("The data loader produced no samples.")
    return averages(sums, sample_count)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_run_dir(config, device, teacher_type, weights):
    name = config["experiment"]["name"]
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = project_path(config["experiment"].get("output_root", "finetune_lut_exp"))
    run_dir = run_dir / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    resolved = dict(config)
    resolved["runtime"] = {
        "device": str(device),
        "run_dir": str(run_dir),
        "effective_loss_weights": weights,
    }
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)
    metadata = {
        "experiment": name,
        "stage": "A0" if teacher_type == "original_mmnet" else "A1",
        "teacher_type": teacher_type,
        "teacher_output": "RGB float tensor [B,3,H,W], nominal [0,1]",
        "student_output": "RGB float tensor [B,3,H,W], unclamped/unbounded",
        "git_commit_at_start": git_commit(),
        "seed": int(config["experiment"].get("seed", 42)),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return run_dir


def log_metrics(writer, split, metrics, epoch):
    for name, value in metrics.items():
        writer.add_scalar(f"{split}/{name}", value, epoch)
    text = ", ".join(f"{name}={value:.6f}" for name, value in metrics.items())
    print(f"{split} epoch {epoch + 1}: {text}")


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["experiment"].get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = loss_weights(config)
    teacher_type, teacher = build_teacher(config, device)
    train_loader, val_loader = build_loaders(config, teacher_type, device)
    lut_model, context = load_student(config, device)

    run_dir = make_run_dir(config, device, teacher_type, weights)
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    structural_loss = fusion_loss().to(device)
    regularizer = TV_4D().to(device)
    optimizer = optim.Adam(
        chain(lut_model.parameters(), context.parameters()),
        lr=float(config["training"]["learning_rate"]),
    )

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Teacher: {teacher_type} (training only, frozen={teacher is not None})")
    print(f"Device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Effective loss weights: {weights}")

    best_val = float("inf")
    best_epoch = None
    epochs = int(config["training"]["epochs"])
    validate_every = int(config["training"].get("validate_every", 10))
    try:
        for epoch in range(epochs):
            train_metrics = run_epoch(
                train_loader,
                lut_model,
                context,
                teacher,
                device,
                structural_loss,
                regularizer,
                weights,
                optimizer=optimizer,
                debug_contract=epoch == 0,
            )
            log_metrics(writer, "train", train_metrics, epoch)

            should_validate = (epoch + 1) % validate_every == 0 or epoch + 1 == epochs
            if should_validate:
                val_metrics = run_epoch(
                    val_loader,
                    lut_model,
                    context,
                    teacher,
                    device,
                    structural_loss,
                    regularizer,
                    weights,
                )
                log_metrics(writer, "val", val_metrics, epoch)
                if val_metrics["total"] < best_val:
                    best_val = val_metrics["total"]
                    best_epoch = epoch + 1
                    save_lut(lut_model, run_dir / "best_lut.npy")
                    save_generator_context(context, run_dir / "best_context.pth")

        save_lut(lut_model, run_dir / "final_lut.npy")
        save_generator_context(context, run_dir / "final_context.pth")
        with open(run_dir / "training_summary.json", "w", encoding="utf-8") as stream:
            json.dump(
                {"best_epoch": best_epoch, "best_val_total": best_val},
                stream,
                indent=2,
            )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
