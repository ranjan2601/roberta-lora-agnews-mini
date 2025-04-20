# roberta-lora-agnews-mini

# RoBERTa + LoRA for AG News Classification (Under 1M Trainable Parameters)

This repository contains the implementation for a Deep Learning mini-project (Spring 2025), which focuses on fine-tuning a RoBERTa model using Low-Rank Adaptation (LoRA) on the AG News dataset. The goal is to maximize classification accuracy while keeping the number of trainable parameters below 1 million.

## 🧠 Project Goal

- Use only **LoRA** to adapt a frozen RoBERTa model
- Ensure **trainable parameters < 1M**
- Achieve the highest possible **test accuracy** on the AG News dataset

## 📦 Dataset

- **AG News** text classification dataset from HuggingFace Datasets.
- 4 Classes: World, Sports, Business, Sci/Tech

## 🛠️ Key Techniques Used

- **LoRA (Low-Rank Adaptation)** using PEFT library
- **RoBERTa-base** architecture (frozen)
- Fine-tuning with **label smoothing**, custom trainer, cosine LR scheduler
- **Custom metrics**: Accuracy, F1-score, styled classification report
- Parameter tracking to ensure < 1M trainable parameters

## 📊 Results

| Metric        | Value     |
|---------------|-----------|
| Test Accuracy | **94%**   |
| Trainable Params | ~950K |
| Epochs        | 5         |
| Batch Size    | 16 |




