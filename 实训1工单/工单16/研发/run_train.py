import traceback, sys, os
os.chdir("/home/swl/LLaMA-Factory")
sys.argv = [
    "train",
    "--model_name_or_path", "/mnt/e/models/Qwen/Qwen2-VL-2B-Instruct",
    "--quantization_bit", "4",
    "--finetuning_type", "lora",
    "--lora_target", "q_proj,v_proj",
    "--lora_rank", "4",
    "--lora_alpha", "8",
    "--lora_dropout", "0.05",
    "--dataset_dir", "/home/swl/LLaMA-Factory/data",
    "--dataset", "imdr_vlm",
    "--template", "qwen2_vl",
    "--cutoff_len", "512",
    "--output_dir", "/mnt/e/桌面/工单/RAG工单16/output",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "16",
    "--num_train_epochs", "1",
    "--learning_rate", "2e-4",
    "--bf16", "true",
    "--gradient_checkpointing", "true",
    "--logging_steps", "5",
    "--image_max_pixels", "65536",
    "--image_min_pixels", "1024",
    "--report_to", "none"
]
try:
    from llamafactory.train.tuner import run_exp
    print("开始训练...", flush=True)
    run_exp()
    print("训练完成!", flush=True)
except Exception as e:
    print(f"错误: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
