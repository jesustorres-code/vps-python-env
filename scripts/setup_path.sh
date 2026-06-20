#!/bin/bash
set -e

echo "Configurando PATH para ~/.local/bin..."

if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

source ~/.bashrc

echo "Configuración completada."
echo "PATH actual:"
echo "$PATH"
