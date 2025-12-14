import numpy as np

ssim = np.array([0.9136, 0.9154, 0.9352, 0.9303])
psnr = np.array([18.4245, 18.8578, 21.9146, 20.3368])
lpips = np.array([0.0906, 0.0888, 0.0773, 0.0734])

print(f"Average SSIM: {ssim.mean():.3f}")
print(f"Average PSNR: {psnr.mean():.1f}")
print(f"Average LPIPS: {lpips.mean():.4f}")