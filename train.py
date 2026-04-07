from data import get_train_dataset
from model import get_model, get_tokenizer
from prompt import format_example
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from config import get_settings, get_lora_config, get_training_config, get_wandb_config
from tracking import init_wandb, finish_wandb


def train():
    settings = get_settings()
    lora_config = get_lora_config()
    training = get_training_config()
    wandb_config = get_wandb_config()

    model = get_model()
    tokenizer = get_tokenizer()
    train_dataset = get_train_dataset()

    dataset = train_dataset.map(format_example)

    init_wandb(wandb_config, training_config=training, settings=settings)

    training_args = SFTConfig(
        output_dir=settings.output_dir,
        num_train_epochs=training.num_epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        lr_scheduler_type=training.lr_scheduler_type,
        warmup_ratio=training.warmup_ratio,
        bf16=training.bf16,
        logging_steps=training.logging_steps,
        save_steps=training.save_steps,
        max_length=settings.max_length,
        dataset_text_field="text",
        packing=False,
        report_to="wandb" if wandb_config.enabled else "none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=lora_config,
    )

    trainer.train()
    finish_wandb()


if __name__ == "__main__":
    train()
