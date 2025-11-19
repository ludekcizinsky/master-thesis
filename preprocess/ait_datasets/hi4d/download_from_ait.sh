#!/bin/bash

set -euo pipefail

links=(
	"https://hi4d.ait.ethz.ch/download.php?dt=def50200cd8c8a592d10547582dca3c5f3a4a2bb9f5563d43d11d265c7644e20b28fa917f14170decab44d9c19b44ad074e5b28d3a662bdc684cb42a8c152e6c19713e62b1d325408823aca0f6245117806c41aab2ef603a3bfb0dd0fb7e917f25171f6968e0c112d8b05cf74d25&file=/pair00_1.tar.gz"
	"https://hi4d.ait.ethz.ch/download.php?dt=def50200cd8c8a592d10547582dca3c5f3a4a2bb9f5563d43d11d265c7644e20b28fa917f14170decab44d9c19b44ad074e5b28d3a662bdc684cb42a8c152e6c19713e62b1d325408823aca0f6245117806c41aab2ef603a3bfb0dd0fb7e917f25171f6968e0c112d8b05cf74d25&file=/pair00_2.tar.gz"
	"https://hi4d.ait.ethz.ch/download.php?dt=def50200cd8c8a592d10547582dca3c5f3a4a2bb9f5563d43d11d265c7644e20b28fa917f14170decab44d9c19b44ad074e5b28d3a662bdc684cb42a8c152e6c19713e62b1d325408823aca0f6245117806c41aab2ef603a3bfb0dd0fb7e917f25171f6968e0c112d8b05cf74d25&file=/pair01.tar.gz"
	"https://hi4d.ait.ethz.ch/download.php?dt=def50200cd8c8a592d10547582dca3c5f3a4a2bb9f5563d43d11d265c7644e20b28fa917f14170decab44d9c19b44ad074e5b28d3a662bdc684cb42a8c152e6c19713e62b1d325408823aca0f6245117806c41aab2ef603a3bfb0dd0fb7e917f25171f6968e0c112d8b05cf74d25&file=/pair15_1.tar.gz"
	"https://hi4d.ait.ethz.ch/download.php?dt=def50200cd8c8a592d10547582dca3c5f3a4a2bb9f5563d43d11d265c7644e20b28fa917f14170decab44d9c19b44ad074e5b28d3a662bdc684cb42a8c152e6c19713e62b1d325408823aca0f6245117806c41aab2ef603a3bfb0dd0fb7e917f25171f6968e0c112d8b05cf74d25&file=/pair15_2.tar.gz"
	"https://hi4d.ait.ethz.ch/download.php?dt=def50200cd8c8a592d10547582dca3c5f3a4a2bb9f5563d43d11d265c7644e20b28fa917f14170decab44d9c19b44ad074e5b28d3a662bdc684cb42a8c152e6c19713e62b1d325408823aca0f6245117806c41aab2ef603a3bfb0dd0fb7e917f25171f6968e0c112d8b05cf74d25&file=/pair16.tar.gz"
	"https://hi4d.ait.ethz.ch/download.php?dt=def50200cd8c8a592d10547582dca3c5f3a4a2bb9f5563d43d11d265c7644e20b28fa917f14170decab44d9c19b44ad074e5b28d3a662bdc684cb42a8c152e6c19713e62b1d325408823aca0f6245117806c41aab2ef603a3bfb0dd0fb7e917f25171f6968e0c112d8b05cf74d25&file=/pair17_1.tar.gz"
	"https://hi4d.ait.ethz.ch/download.php?dt=def50200cd8c8a592d10547582dca3c5f3a4a2bb9f5563d43d11d265c7644e20b28fa917f14170decab44d9c19b44ad074e5b28d3a662bdc684cb42a8c152e6c19713e62b1d325408823aca0f6245117806c41aab2ef603a3bfb0dd0fb7e917f25171f6968e0c112d8b05cf74d25&file=/pair17_2.tar.gz"
	"https://hi4d.ait.ethz.ch/download.php?dt=def50200cd8c8a592d10547582dca3c5f3a4a2bb9f5563d43d11d265c7644e20b28fa917f14170decab44d9c19b44ad074e5b28d3a662bdc684cb42a8c152e6c19713e62b1d325408823aca0f6245117806c41aab2ef603a3bfb0dd0fb7e917f25171f6968e0c112d8b05cf74d25&file=/pair19_1.tar.gz"
	"https://hi4d.ait.ethz.ch/download.php?dt=def50200cd8c8a592d10547582dca3c5f3a4a2bb9f5563d43d11d265c7644e20b28fa917f14170decab44d9c19b44ad074e5b28d3a662bdc684cb42a8c152e6c19713e62b1d325408823aca0f6245117806c41aab2ef603a3bfb0dd0fb7e917f25171f6968e0c112d8b05cf74d25&file=/pair19_2.tar.gz"
)

for url in "${links[@]}"; do
  filename=$(echo "$url" | sed -n 's/.*file=\/\([^&]*\)/\1/p')
  echo "Downloading $filename ..."
  wget -c --tries=5 --waitretry=5 --retry-connrefused -O "$filename" "$url"
  echo "✅ Finished: $filename"
  echo
done

echo "🎉 All downloads completed."
