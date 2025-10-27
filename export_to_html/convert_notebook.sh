#!/bin/bash
# Convert notebook to HTML with images embedded and move to project root
# Execute this script from the notebook folder or ajuste NOTEBOOK_PATH

NOTEBOOK="../notebooks/1.0.tgs.cookies_cats_ab_test.ipynb"
OUTPUT="index.html"
PROJECT_ROOT="../"  

# --- Convert notebook to HTML ---
jupyter nbconvert \
    --to html \
    --execute \
    --embed-images \
    "$NOTEBOOK" \
    --output "index.html" \
    --output-dir="$PROJECT_ROOT"


echo "Notebook converted and moved to project root: $PROJECT_ROOT/$OUTPUT"