import argparse
import os
import time

import numpy as np
import torch
from torchvision.transforms import ToPILImage

import transforms as T
from data.simple_dataset import SimpleDataSet
from scripts.calculate import (
    Generator_for_info,
    apply_fusion_4d_with_interpolation,
    load_lookup_table,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LUT-Fuse inference and save benchmark metrics."
    )
    parser.add_argument(
        "--lut_path",
        default="ckpts/fine_tuned_lut.npy",
        help="Path to the fusion LUT .npy checkpoint.",
    )
    parser.add_argument(
        "--context_path",
        default="ckpts/generator_context.pth",
        help="Path to the context generator checkpoint.",
    )
    parser.add_argument("--visible_dir", required=True)
    parser.add_argument("--infrared_dir", required=True)
    parser.add_argument("--save_dir", default="fusion_results")
    parser.add_argument("--num_workers", type=int, default=1)
    return parser.parse_args()


def sync_cuda_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_parameters(module):
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return int(total), int(trainable)


def format_parameter_count(count):
    return f"{int(count):,} ({float(count) / 1_000_000.0:.3f} M)"


def format_mib(num_bytes):
    if num_bytes is None:
        return "N/A (CUDA not available)"
    return f"{float(num_bytes) / (1024.0 ** 2):.2f} MiB"


def format_input_size_counts(input_size_counts):
    if not input_size_counts:
        return "None"
    return ", ".join(
        f"{height}x{width} ({count} image(s))"
        for (height, width), count in sorted(input_size_counts.items())
    )


def move_batch_to_device(data, device):
    visible, infrared, names = data
    non_blocking = device.type == "cuda"
    visible = visible.to(device, non_blocking=non_blocking)
    infrared = infrared.to(device, non_blocking=non_blocking)

    if visible.shape[-2:] != infrared.shape[-2:]:
        raise ValueError(
            "Visible/infrared spatial size mismatch: "
            f"visible={tuple(visible.shape)}, infrared={tuple(infrared.shape)}"
        )

    return visible, infrared, names


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lut = load_lookup_table(args.lut_path, device=device)
    if lut is None:
        raise RuntimeError(f"Failed to load LUT checkpoint: {args.lut_path}")

    get_context = Generator_for_info().to(device)
    get_context.load_state_dict(torch.load(args.context_path, map_location=device))
    get_context.eval()

    context_total_parameters, context_trainable_parameters = count_parameters(
        get_context
    )
    lut_shape = tuple(int(value) for value in lut.shape)
    lut_entries = int(lut.numel())
    total_inference_values = context_total_parameters + lut_entries

    print("----------------------------------------")
    print(f"Device: {device}")
    print(f"Context generator parameters: {format_parameter_count(context_total_parameters)}")
    print(f"LUT shape: {lut_shape}")
    print(f"LUT entries: {format_parameter_count(lut_entries)}")
    print(f"Total inference learned values: {format_parameter_count(total_inference_values)}")
    print("----------------------------------------")

    data_transform = {
        "val": T.Compose([T.Resize_16(), T.ToTensor()]),
    }

    val_dataset = SimpleDataSet(
        visible_path=args.visible_dir,
        infrared_path=args.infrared_dir,
        phase="val",
        transform=data_transform["val"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
        collate_fn=val_dataset.collate_fn,
    )

    if len(val_dataset.visible_files) != len(val_dataset.infrared_files):
        raise ValueError(
            "The number of visible and infrared images does not match: "
            f"visible={len(val_dataset.visible_files)}, "
            f"infrared={len(val_dataset.infrared_files)}"
        )
    if len(val_dataset) == 0:
        raise ValueError("No paired input images were found.")
    warmup_data = next(iter(val_loader))
    warmup_visible, warmup_infrared, _ = move_batch_to_device(
        warmup_data,
        device,
    )
    print("Running one warm-up inference (excluded from benchmark)...")
    with torch.no_grad():
        sync_cuda_if_needed(device)
        _ = apply_fusion_4d_with_interpolation(
            warmup_visible * 255.0,
            warmup_infrared * 255.0,
            lut,
            get_context,
        )
        sync_cuda_if_needed(device)
    print("Warm-up finished.")

    inference_times = []
    input_size_counts = {}
    peak_vram_allocated_bytes = None
    peak_vram_reserved_bytes = None

    with torch.no_grad():
        for step, data in enumerate(val_loader):
            visible, infrared, names = move_batch_to_device(data, device)
            input_height, input_width = visible.shape[-2:]

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            sync_cuda_if_needed(device)
            start_time = time.perf_counter()
            outputs = apply_fusion_4d_with_interpolation(
                visible * 255.0,
                infrared * 255.0,
                lut,
                get_context,
            )
            sync_cuda_if_needed(device)
            elapsed_time = time.perf_counter() - start_time
            inference_times.append(elapsed_time)

            input_size_key = (int(input_height), int(input_width))
            input_size_counts[input_size_key] = (
                input_size_counts.get(input_size_key, 0) + 1
            )

            memory_text = ""
            if device.type == "cuda":
                image_peak_allocated = torch.cuda.max_memory_allocated(device)
                image_peak_reserved = torch.cuda.max_memory_reserved(device)
                peak_vram_allocated_bytes = max(
                    peak_vram_allocated_bytes or 0,
                    image_peak_allocated,
                )
                peak_vram_reserved_bytes = max(
                    peak_vram_reserved_bytes or 0,
                    image_peak_reserved,
                )
                memory_text = (
                    f" | peak allocated: {format_mib(image_peak_allocated)}"
                    f" | reserved: {format_mib(image_peak_reserved)}"
                )

            output_name = names[0]
            if not os.path.splitext(output_name)[1]:
                output_name += ".png"
            save_path = os.path.join(args.save_dir, output_name)
            fusion_result = outputs.squeeze(0).clamp(0, 1).cpu()
            ToPILImage()(fusion_result).save(save_path)

            print(
                f"[OK] {step + 1}/{len(val_dataset)} {output_name}"
                f" | input: {input_height}x{input_width}"
                f" | latency: {elapsed_time * 1000.0:.3f} ms"
                f"{memory_text}"
            )

    measured_images = len(inference_times)
    total_time_sec = float(np.sum(inference_times, dtype=np.float64))
    average_time_sec = total_time_sec / measured_images
    std_time_sec = float(np.std(inference_times, dtype=np.float64))
    fps = measured_images / total_time_sec if total_time_sec > 0 else 0.0

    metrics_path = os.path.join(args.save_dir, "benchmark_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("[LUT-Fuse inference benchmark]\n")
        file.write(f"Device: {device}\n")
        file.write("Batch size: 1\n")
        file.write(f"Measured images: {measured_images}\n")
        file.write(f"Input image sizes (HxW): {format_input_size_counts(input_size_counts)}\n")
        file.write("\n[Latency and throughput]\n")
        file.write(
            "Scope: apply_fusion_4d_with_interpolation only; "
            "excludes data loading, resizing, and image saving\n"
        )
        file.write("Warm-up: one pass excluded from all measurements\n")
        file.write(f"Total inference time: {total_time_sec:.6f} sec\n")
        file.write(f"Average inference latency: {average_time_sec:.6f} sec/image\n")
        file.write(f"Average inference latency: {average_time_sec * 1000.0:.3f} ms/image\n")
        file.write(f"Latency standard deviation: {std_time_sec * 1000.0:.3f} ms\n")
        file.write(f"Throughput: {fps:.3f} FPS\n")
        file.write("\n[Parameters]\n")
        file.write(f"Context generator parameters: {format_parameter_count(context_total_parameters)}\n")
        file.write(f"Context generator trainable parameters: {format_parameter_count(context_trainable_parameters)}\n")
        file.write(f"LUT shape: {'x'.join(str(value) for value in lut_shape)}\n")
        file.write(f"LUT entries: {format_parameter_count(lut_entries)}\n")
        file.write(f"Total inference learned values: {format_parameter_count(total_inference_values)}\n")
        file.write("\n[CUDA VRAM]\n")
        file.write(f"Peak memory allocated: {format_mib(peak_vram_allocated_bytes)}\n")
        file.write(f"Peak memory reserved: {format_mib(peak_vram_reserved_bytes)}\n")
        file.write(
            "VRAM note: maximum across measured images, including "
            "checkpoints and input tensors.\n"
        )

    print("----------------------------------------")
    print("LUT-Fuse inference finished")
    print(f"Measured images: {measured_images}")
    print(f"Input image sizes (HxW): {format_input_size_counts(input_size_counts)}")
    print(f"Total inference time: {total_time_sec:.6f} sec")
    print(f"Average inference latency: {average_time_sec * 1000.0:.3f} ms/image")
    print(f"Latency standard deviation: {std_time_sec * 1000.0:.3f} ms")
    print(f"Throughput: {fps:.3f} FPS")
    print(f"Total inference learned values: {format_parameter_count(total_inference_values)}")
    print(f"Peak CUDA memory allocated: {format_mib(peak_vram_allocated_bytes)}")
    print(f"Peak CUDA memory reserved: {format_mib(peak_vram_reserved_bytes)}")
    print(f"Benchmark metrics: {metrics_path}")
    print("----------------------------------------")

if __name__ == "__main__":
    main(parse_args())

