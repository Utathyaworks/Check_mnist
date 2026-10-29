# Check_mnist

MNIST CNN classifier (PyTorch). Trains on GPU, logs per-epoch metrics, and saves
loss/accuracy curves, a confusion matrix and sample predictions to `results/`.

Run: `python mnist.py` (needs torch, torchvision, matplotlib).

Result on an A100-40GB: 99.00% test accuracy after 8 epochs (about 9 s of training).
