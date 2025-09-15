# RARE2025_MEDAI

We propose a binary classification framework for endoscopic neoplasia detection using a ResNet50 backbone with GastroNet pretraining. To handle class imbalance, we use Balanced MixUp for stable decision boundaries. For calibrated probabilities, Platt scaling is applied, and EMA further improves generalization. Training uses cosine learning rate scheduling, warm-up, and AdamW optimization, achieving strong performance on the RARE25 dataset.

To reproduce checkpoints, simply run: `run_train.sh`
