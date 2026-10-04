"""LoRA / QLoRA fine-tune of an open model with Unsloth. GPU machine only.

Examples:
  python training/train_lora.py --data data/train/persona_sft.jsonl --base Qwen/Qwen3.8-27B --out runs/persona-qwen
  python training/train_lora.py --data data/train/persona_sft.jsonl --base sarvamai/sarvam-30b --out runs/persona-sarvam
  python training/train_lora.py --data data/train/ranker_train.jsonl --base Qwen/Qwen3.5-4B --out runs/ranker-small

Serve the result with vLLM:
  vllm serve Qwen/Qwen3.8-27B --enable-lora --lora-modules vsp=runs/persona-qwen/adapter
then enable "vsp-tuned" in config/models.yaml. It only goes live after training/eval_gate.py passes.
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--gguf", action="store_true", help="also export a q4_k_m GGUF for Ollama / llama.cpp")
    a = ap.parse_args()

    from unsloth import FastLanguageModel  # import first so Unsloth can patch trl/transformers
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = FastLanguageModel.from_pretrained(model_name=a.base, max_seq_length=a.max_seq, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=a.r, lora_alpha=a.r, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)

    ds = load_dataset("json", data_files=a.data, split="train")
    ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)})

    cfg = SFTConfig(output_dir=a.out, dataset_text_field="text", per_device_train_batch_size=a.batch,
                    gradient_accumulation_steps=a.grad_accum, num_train_epochs=a.epochs, learning_rate=a.lr,
                    warmup_ratio=0.03, lr_scheduler_type="linear", optim="adamw_8bit", logging_steps=10,
                    save_strategy="epoch", seed=3407, report_to="none")
    try:
        trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, args=cfg)
    except TypeError:  # older trl
        trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=cfg)
    trainer.train()
    model.save_pretrained(f"{a.out}/adapter")
    tok.save_pretrained(f"{a.out}/adapter")
    if a.gguf:
        model.save_pretrained_gguf(f"{a.out}/gguf", tok, quantization_method="q4_k_m")
    print(f"Saved adapter to {a.out}/adapter. Next: serve it, then run training/eval_gate.py before using it.")


if __name__ == "__main__":
    main()
