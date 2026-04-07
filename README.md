## Finetuning [Gemma 4](https://huggingface.co/blog/gemma4) base model using PEFT/LoRA on bengali dataset

![alt text](image.png)

This repo performs a simple peft process on a base llm. The following environment variables should be set in a `.env` file in the root of the project.

- MODEL_ID=_Base model from huggingface, I used `gemma-4-E2B` for testing_
- DATASET_ID=_Name of the dataset with instruction and response pairs. `iamshnoo/alpaca-cleaned-bengali` is a good starting point._
- OUTPUT_DIR=_Location of the saved fine-tuned model_

---

The training script relies on Supervised fine tuning (SFT) using the `trl` python library.

