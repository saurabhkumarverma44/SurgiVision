import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    test_tensor = torch.randn(1, 1, 64, 64, 64, device=device)
    print("✅ GPU Test Success!")
    print(f"GPU Memory Used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"RTX 4050 Utilization: {torch.cuda.memory_reserved()/6e9*100:.1f}%")
    print(f"Tensor shape: {test_tensor.shape}")
else:
    print("❌ CUDA still not available")
