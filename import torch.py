import torch

# Create two tensors with dimensions (2, 3, 4) filled with random values.
tensor1 = torch.randn(2, 3, 4)
tensor2 = torch.randn(2, 3, 4)

A,B,C=tensor2.shape
print("Tensor 1:\n", tensor1)
print("\nTensor 2:\n", tensor2)

print("Tensor 1:\n", tensor1.shape)
print("\nTensor 2:\n", tensor2.shape)

tensor3=tensor2.reshape(A,C,B)
tensor4=tensor2.transpose(1,2)
print(f"multiplication 1: {torch.concat(tensors=[tensor1,tensor2])}")
print(f"multiplication 1: {torch.concat(tensors=[tensor1,tensor2]).shape}")
print(f"multiplication 2: {torch.concat(tensors=[tensor3,tensor4])}")
print(f"multiplication 2: {torch.concat(tensors=[tensor3,tensor4]).shape}")
print(f"multiplication 3: {torch.concat(tensors=[tensor3,tensor2])}")
print(f"multiplication 3: {torch.concat(tensors=[tensor3,tensor2]).shape}")
print(f"multiplication 4: {torch.concat(tensors=[tensor2,tensor3])}")
print(f"multiplication 4: {torch.concat(tensors=[tensor2,tensor3]).shape}")

