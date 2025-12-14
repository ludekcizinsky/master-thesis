import numpy as np

# - zero shot lhm
# ssim = np.array([0.9136, 0.9154, 0.9352, 0.9303])
# psnr = np.array([18.4245, 18.8578, 21.9146, 20.3368])
# lpips = np.array([0.0906, 0.0888, 0.0773, 0.0734])

# - zero shot lhm + difix refine
ssim = np.array([0.7637, 0.7352, 0.7890, 0.7445])
psnr = np.array([17.7862, 18.2688, 21.3553, 19.8489])
lpips = np.array([0.0812, 0.0843, 0.0703, 0.0728])

print(f"Average SSIM: {ssim.mean():.3f}")
print(f"Average PSNR: {psnr.mean():.1f}")
print(f"Average LPIPS: {lpips.mean():.4f}")