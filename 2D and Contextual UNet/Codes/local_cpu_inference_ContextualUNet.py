"""
Author: Roa'a Khaled
Affiliation: Department of Computer Engineering, Department of Condensed Matter Physics,
             University of Cádiz, , Puerto Real, 11519, Cádiz, Spain.
Email: roaa.khaled@gm.uca.es
Date: 2025-07-27
Project: This work is part of a predoctoral research and part of PARENT project that has received funding
         from the EU’s Horizon 2020 research and innovation program under the MSCA – ITN 2020, GA No 956394.

Description:
    This code uses a previously trained Contextual UNet model to segment testing/inference data, the model was trained
    for Total Brains segmentation from cranial Ultrasound images of premature born infants.

Usage:
    python local_cpu_inference_nnUNet.py

Notes:
    - for environment requirements check
"""

import os
import time
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import psutil
import tracemalloc
from skimage.transform import resize
from ptflops import get_model_complexity_info
from UNet2D_pyImageSearch import local_inference_config
from collections import defaultdict
from openpyxl import load_workbook

# Force CPU usage
DEVICE = torch.device("cpu")

# Load model on CPU
print("[INFO] Loading model on CPU...")
model = torch.load(local_inference_config.MODEL_PATH, map_location=DEVICE).to(DEVICE)
model.eval()

# Show system resource info before running
virtual_mem = psutil.virtual_memory()
cpu_load = psutil.cpu_percent(interval=1)
print("\n🔍 System Resource Snapshot BEFORE inference:")
print(f"Available RAM: {virtual_mem.available / (1024 ** 2):.2f} MB")
print(f"Total RAM: {virtual_mem.total / (1024 ** 2):.2f} MB")
print(f"CPU Usage: {cpu_load:.1f}%\n")

# === Calculate FLOPs and number of parameters ===
def profile_model(model):
    # Verify expected input channels
    first_conv = next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    print(f"👀 Model expects input channels = {first_conv.in_channels}")

    def input_constructor(input_res):
        # input_res is (C, W, H)
        return torch.zeros(1, *input_res).to(DEVICE)

    macs, params = get_model_complexity_info(
        model,
        (3,
         local_inference_config.INPUT_IMAGE_WIDTH,
         local_inference_config.INPUT_IMAGE_HEIGHT),
        as_strings=False, print_per_layer_stat=False,
        verbose=False,
        input_constructor=input_constructor
    )
    return round(macs / 1e9, 2), round(params / 1e6, 2) # in GMac and M

flops_gmac, params_m = profile_model(model)
profile_excel_path = os.path.join(local_inference_config.PREDICTIONS_PATH, "cpu_inference_model_profile.xlsx")
df_profile = pd.DataFrame([{
    "Model": "2D UNet",
    "FLOPs (GMacs)": flops_gmac,
    "Parameters (Millions)": params_m
}])
df_profile.to_excel(profile_excel_path, index=False)

# === Load image paths ===
imagePaths = open(local_inference_config.TESTING_IMAGES_PATHS).read().strip().split("\n")
# === Load 3 consecutive slices and handling biundaries ===
def load_consecutive_slices(folder, base_name, slice_num):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".nii.gz")])
    max_idx = len(files) - 1
    slices = []
    for offset in (-1, 0, 1):
        idx = slice_num + offset
        if idx < 0:
            idx = 0
        elif idx > max_idx:
            idx = max_idx
        slice_path = os.path.join(folder, files[idx])
        img = nib.load(slice_path)
        img_arr = img.get_fdata().astype(np.float32)
        img_arr = resize(img_arr,
                         (local_inference_config.INPUT_IMAGE_HEIGHT,
                          local_inference_config.INPUT_IMAGE_WIDTH),
                         order=3, anti_aliasing=True)
        slices.append(img_arr)
    return slices

# Group paths by 3D scan (patient/date combo)
grouped_paths = defaultdict(list)
for path in imagePaths:
    patient = os.path.basename(os.path.dirname(os.path.dirname(path)))
    date = os.path.basename(os.path.dirname(path))
    grouped_paths[(patient, date)].append(path)

# Prepare Excel output paths
excel_2d_path = os.path.join(local_inference_config.PREDICTIONS_PATH, "cpu_inference_2d_slices.xlsx")
excel_3d_path = os.path.join(local_inference_config.PREDICTIONS_PATH, "cpu_inference_3d_scans.xlsx")

patient_sheets = defaultdict(list)
all_3d_entries = []
peak_cpu_memory_all = []
peak_tracemalloc_all = []

tracemalloc.start()  # Start tracking Python-level allocations

for (patient, date), paths in grouped_paths.items():
    total_time_3d = 0.0
    mem_usages_2d = []
    trace_usages_2d = []
    times_2d = []
    peak_mem_this_scan = 0.0
    peak_trace_this_scan = 0.0

    for path in paths:
        slice_num = int(os.path.basename(path).split('_')[-1].split('.')[0])
        base_name = '_'.join(os.path.basename(path).split('_')[:-1])
        folder = os.path.dirname(path)

        # Load three slices: prev, current, next
        slices = load_consecutive_slices(folder, base_name, slice_num)

        # Stack into shape (1, 3, H, W)
        image_arr = np.stack(slices, axis=0)
        image_arr = np.expand_dims(image_arr, axis=0)

        image_tensor = torch.from_numpy(image_arr).to(DEVICE)

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss

        t0 = time.time()
        with torch.no_grad():
            _ = model(image_tensor)
        t1 = time.time()

        mem_after = process.memory_info().rss
        mem_usage = max(mem_before, mem_after) / 1024**2  # in MB
        _, trace_peak = tracemalloc.get_traced_memory()
        trace_usage = trace_peak / 1024**2  # in MB

        peak_mem_this_scan = max(peak_mem_this_scan, mem_usage)
        peak_trace_this_scan = max(peak_trace_this_scan, trace_usage)

        inference_time = round(t1 - t0, 4)
        times_2d.append(inference_time)
        mem_usages_2d.append(mem_usage)
        trace_usages_2d.append(trace_usage)
        total_time_3d += inference_time

        patient_sheets[patient].append({
            "Patient ID": patient,
            "Scan Date": date,
            "Scan Name": os.path.basename(path),
            "Inference Time (s)": inference_time,
            "CPU Memory (psutil MB)": round(mem_usage, 2),
            "CPU Memory (tracemalloc MB)": round(trace_usage, 2)
        })

        del image_tensor
        torch.cuda.empty_cache()

    avg_time_2d = round(np.mean(times_2d), 4)
    avg_mem_2d = round(np.mean(mem_usages_2d), 2)
    avg_trace_2d = round(np.mean(trace_usages_2d), 2)
    all_3d_entries.append({
        "Patient ID": patient,
        "Scan Date": date,
        "#Slices": len(paths),
        "Total Time (3D) (s)": round(total_time_3d, 4),
        "Avg Time per 2D Slice (s)": avg_time_2d,
        "Avg CPU Memory per 2D (psutil MB)": avg_mem_2d,
        "Avg CPU Memory per 2D (tracemalloc MB)": avg_trace_2d,
        "Peak CPU Memory (psutil MB)": round(peak_mem_this_scan, 2),
        "Peak CPU Memory (tracemalloc MB)": round(peak_trace_this_scan, 2)
    })

    peak_cpu_memory_all.append(peak_mem_this_scan)
    peak_tracemalloc_all.append(peak_trace_this_scan)

# Save 2D results per patient to Excel
with pd.ExcelWriter(excel_2d_path, engine='openpyxl') as writer:
    for patient, records in patient_sheets.items():
        pd.DataFrame(records).to_excel(writer, sheet_name=patient, index=False)
    # Add sheet with global averages
    all_rows = [row for records in patient_sheets.values() for row in records]
    df_all = pd.DataFrame(all_rows)
    avg_2d = pd.DataFrame([{
        "Avg Inference Time (s)": round(df_all["Inference Time (s)"].mean(), 4),
        "Avg CPU Memory (psutil MB)": round(df_all["CPU Memory (psutil MB)"].mean(), 2),
        "Avg CPU Memory (tracemalloc MB)": round(df_all["CPU Memory (tracemalloc MB)"].mean(), 2)
    }])
    avg_2d.to_excel(writer, sheet_name="Averages", index=False)

# Save 3D scan results to Excel
with pd.ExcelWriter(excel_3d_path, engine='openpyxl') as writer:
    df_3d = pd.DataFrame(all_3d_entries)
    df_3d.to_excel(writer, sheet_name="Per 3D Scan", index=False)
    avg_3d = pd.DataFrame([{
        "Avg Total Time (3D) (s)": round(df_3d["Total Time (3D) (s)"].mean(), 4),
        "Avg Time per 2D Slice (s)": round(df_3d["Avg Time per 2D Slice (s)"].mean(), 4),
        "Avg CPU Memory per 2D (psutil MB)": round(df_3d["Avg CPU Memory per 2D (psutil MB)"].mean(), 2),
        "Avg CPU Memory per 2D (tracemalloc MB)": round(df_3d["Avg CPU Memory per 2D (tracemalloc MB)"].mean(), 2),
        "Avg Peak CPU Memory (psutil MB)": round(np.mean(peak_cpu_memory_all), 2),
        "Avg Peak CPU Memory (tracemalloc MB)": round(np.mean(peak_tracemalloc_all), 2)
    }])
    avg_3d.to_excel(writer, sheet_name="Averages", index=False)

print(f"✅ Inference results saved to:\n  - 2D Slices: {excel_2d_path}\n  - 3D Scans : {excel_3d_path}\n  - Model Profile: {profile_excel_path}")
