"""
Author: Roa'a Khaled
Affiliation: Department of Computer Engineering, Department of Condensed Matter Physics,
             University of Cádiz, , Puerto Real, 11519, Cádiz, Spain.
Email: roaa.khaled@gm.uca.es
Date: 2025-07-27
Project: This work is part of a predoctoral research and part of PARENT project that has received funding
         from the EU’s Horizon 2020 research and innovation program under the MSCA – ITN 2020, GA No 956394.

Description:
    This code uses a previously trained nnUNet model (2d or 3d) to segment testing/inference data on a low-resource
     machine (CPU only), the model was trained for Total Brains segmentation from cranial Ultrasound images of
     premature born infants.

Usage:
    python local_cpu_inference_nnUNet.py

Notes:
    - for environment requirements check
    - you need to prepare your data in the nnUNet fprmat and do preprocessing first using nnUNetv2_plan_and_preprocess,
    check: (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md)
"""

import os
# --- Set environment variables (MUST be done before importing nnUNetTrainer or running inference) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU usage
os.environ["nnUNet_raw"] = "path/to/your/raw/data" # --> data in nnUnet format, check:
# (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)
os.environ["nnUNet_preprocessed"] = "path/to/your/preprocessed/data" # --> produced by running:
# nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
os.environ["nnUNet_results"] = "path/to/trained/nnUNet" # ex. "Total-Brain-Segmentation-from-cUS\nnUNet\nnUNet_results"

import time
import tracemalloc
import torch
import psutil
import openpyxl
import shutil
from ptflops import get_model_complexity_info
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# --- Paths and config ---
task = "Dataset165_NeonatalUltrasoundTB"
fold = 0
config = "3d_fullres" # change to "2d" if you want to use the 2d nnUNet trained model
trainer_name = "nnUNetTrainer"
plans_identifier = "nnUNetPlans"

# path to testing/inference data folder (in nnUNet format)
input_dir = os.path.join(os.environ["nnUNet_raw"], task, "imagesTs")
# path to output segmentations folder
output_dir = "path/to/outputs/folder"
# path to output xlsx file to save inference total time and memory usage
output_excel = "path/to/inference/profiling/file.xlsx"

if __name__ == "__main__":
    # --- Clean/create output dir ---
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # --- Load predictor ---
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=torch.device("cpu"),
        verbose=True
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=os.path.join(os.environ["nnUNet_results"], task, f"{trainer_name}__{plans_identifier}__{config}"),
        use_folds=(fold,),
        checkpoint_name="checkpoint_final.pth"
    )

    # --- Run inference ---
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    start_time = time.time()

    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_dir,
        output_folder_or_list_of_truncated_output_files=output_dir,
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1
    )

    end_time = time.time()
    mem_after = process.memory_info().rss
    peak_cpu_mem_psutil = max(mem_before, mem_after) / 1024**2
    _, peak_cpu_mem_trace = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Save results to Excel ---
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Inference time and memory"
    ws.append(["Model", "Peak CPU Mem (psutil MB)", "Peak CPU Mem (tracemalloc MB)", "Total Inference Time (s)"])
    ws.append([
        config,
        flops_gmac,
        params_m,
        f"{peak_cpu_mem_psutil:.2f}",
        f"{peak_cpu_mem_trace / 1024**2:.2f}",
        f"{end_time - start_time:.2f}"
    ])
    wb.save(output_excel)

    print(f"✅ Inference complete! Predictions saved to: {output_dir}")
    print(f"📊 Profiling results saved to: {output_excel}")
