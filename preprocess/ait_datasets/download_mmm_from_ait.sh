#!/bin/bash

set -euo pipefail

links=(
    "https://multiply.ait.ethz.ch/download.php?dt=def502009760e8207c9a1974b0d839b1c602ac4242f652ce6a28474bb900f06a8dda262bdd1afb728ed973de94efebd08484b23967f4d5f20ee80b5ea942559b72145bb5355e4501cb4dbd544e139f0cd108df255167bb0a1044690f89e2b97f0af5242f7d5453c057ae195e011dbf8ee0e2&file=/MMM/cheer.zip"
    "https://multiply.ait.ethz.ch/download.php?dt=def502009760e8207c9a1974b0d839b1c602ac4242f652ce6a28474bb900f06a8dda262bdd1afb728ed973de94efebd08484b23967f4d5f20ee80b5ea942559b72145bb5355e4501cb4dbd544e139f0cd108df255167bb0a1044690f89e2b97f0af5242f7d5453c057ae195e011dbf8ee0e2&file=/MMM/dance.zip"
    "https://multiply.ait.ethz.ch/download.php?dt=def502009760e8207c9a1974b0d839b1c602ac4242f652ce6a28474bb900f06a8dda262bdd1afb728ed973de94efebd08484b23967f4d5f20ee80b5ea942559b72145bb5355e4501cb4dbd544e139f0cd108df255167bb0a1044690f89e2b97f0af5242f7d5453c057ae195e011dbf8ee0e2&file=/MMM/hug.zip"
    "https://multiply.ait.ethz.ch/download.php?dt=def502009760e8207c9a1974b0d839b1c602ac4242f652ce6a28474bb900f06a8dda262bdd1afb728ed973de94efebd08484b23967f4d5f20ee80b5ea942559b72145bb5355e4501cb4dbd544e139f0cd108df255167bb0a1044690f89e2b97f0af5242f7d5453c057ae195e011dbf8ee0e2&file=/MMM/lift.zip"
    "https://multiply.ait.ethz.ch/download.php?dt=def502009760e8207c9a1974b0d839b1c602ac4242f652ce6a28474bb900f06a8dda262bdd1afb728ed973de94efebd08484b23967f4d5f20ee80b5ea942559b72145bb5355e4501cb4dbd544e139f0cd108df255167bb0a1044690f89e2b97f0af5242f7d5453c057ae195e011dbf8ee0e2&file=/MMM/selfie.zip"
    "https://multiply.ait.ethz.ch/download.php?dt=def502009760e8207c9a1974b0d839b1c602ac4242f652ce6a28474bb900f06a8dda262bdd1afb728ed973de94efebd08484b23967f4d5f20ee80b5ea942559b72145bb5355e4501cb4dbd544e139f0cd108df255167bb0a1044690f89e2b97f0af5242f7d5453c057ae195e011dbf8ee0e2&file=/MMM/walkdance.zip"
)

for url in "${links[@]}"; do
  filename=$(echo "$url" | sed -n 's/.*file=\/\([^&]*\)/\1/p')
  dir=$(dirname "$filename")
  if [ "$dir" != "." ]; then
    mkdir -p "$dir"
  fi
  echo "Downloading $filename ..."
  wget -c --tries=5 --waitretry=5 --retry-connrefused -O "$filename" "$url"
  echo "✅ Finished: $filename"
  echo
done

echo "🎉 All downloads completed."
