#!/usr/bin/env bash
set -euo pipefail

# Point to the images root for the viewer
export BDD_IMG_ROOT="assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k"

streamlit run src/vis/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
