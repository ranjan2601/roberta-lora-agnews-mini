# roberta-lora-agnews-mini

# RoBERTa + LoRA for AG News Classification (Under 1M Trainable Parameters)

This repository contains the implementation for a Deep Learning mini-project (Spring 2025), which focuses on fine-tuning a RoBERTa model using Low-Rank Adaptation (LoRA) on the AG News dataset. The goal is to maximize classification accuracy while keeping the number of trainable parameters below 1 million.

## Project Goal

- Use only **LoRA** to adapt a frozen RoBERTa model
- Ensure **trainable parameters < 1M**
- Achieve the highest possible **test accuracy** on the AG News dataset

## Dataset

- **AG News** text classification dataset from HuggingFace Datasets.
- 4 Classes: World, Sports, Business, Sci/Tech

## Key Techniques Used

- **LoRA (Low-Rank Adaptation)** using PEFT library
- **RoBERTa-base** architecture (frozen)
- Fine-tuning with **label smoothing**, custom trainer, cosine LR scheduler
- **Custom metrics**: Accuracy, F1-score, styled classification report

## Results

### Final Model Configuration

| Parameter            | Value                                   |
|----------------------|-----------------------------------------|
| `r` (LoRA rank)      | 4                                       |
| `lora_alpha`         | 16                                      |
| `lora_dropout`       | 0.0                                     |
| `learning_rate`      | 4e-4                                    |
| `loss` function      | CrossEntropy with label smoothing = 0.1 |
| Optimizer            | AdamW                                   |
| Scheduler            | Cosine                                  |
| Epochs               | 5                                       |
| Trainable Parameters | 741124                                  |

---

### Final Model Performance

| Metric            | Value   |
|------------------|---------|
| Eval Accuracy     | **94.71%** |
| Eval F1-score     | *95%* |
| Eval Loss         | *0.45* |

---

### Comparison: Model Variants & Hyperparameter Tuning

| Run | `r` | `alpha` | Dropout | LR    | Accuracy | Parameters|
|-----|-----|---------|---------|-------|----------|-----------|
| 1   | 4   | 16      | 0.0     | 4e-4  | 94.71%   | 741124    |
| 2   | 4   | 16      | 0.1     | 3e-4  | 94.5%    | 741124    |
| 3   | 4   | 16      | 0.25    | 3e-4  | 93.9%    | 741124    |
| 4   | 8   | 32      | 0.01    | 5e-4  | 94.1%    | 888580    |
| 5   | 8   | 32      | 0.25    | 3e-4  | 94.2%    | 888580    |
| 6   | 8   | 32      | 0.05    | 5e-4  | 93.8%    | 888580    |
| 7   | 11  | 32      | 0.1     | 4e-4  | 94.6%    | 999172    |

> We observed that a lower LoRA rank (`r=4`) paired with a moderate alpha (`alpha=16`) offered the best balance between performance and parameter efficiency. Higher dropout helped mitigate overfitting in deeper configurations.


## Contributions
- **Ranjan (sp8171@nyu.edu)**
- **Nishanth (nk3968@nyu.edu)**
- **Shreyas (sb9855@nyu.edu)**





