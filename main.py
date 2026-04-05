from data import get_train_dataset
from model import get_model, get_tokenizer
from prompt import format_example
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from config import get_settings, get_lora_config


if __name__ == "__main__":
    settings = get_settings()
    lora_config = get_lora_config()
    model = get_model()
    tokenizer = get_tokenizer()
    train_dataset = get_train_dataset()

    dataset = train_dataset.map(format_example)

    training_args = SFTConfig(
        output_dir=settings.output_dir,
        num_train_epochs=3,            # 1 epoch is enough for learning
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, # effective batch = 2 * 4 = 8
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,                     # use fp16=True if your GPU doesn't support bf16
        logging_steps=25,
        save_steps=200,
        max_length=settings.max_length,
        dataset_text_field="text",     # the column we formatted above
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=lora_config
    )

    trainer.train()

