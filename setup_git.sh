#!/bin/bash
# setup_git.sh
# Run this from /root/Dreamwalk to initialise and push to your GitHub repo
# Usage: bash setup_git.sh

# ── CONFIG — change these ─────────────────────────────────────────────────────
REPO_URL="https://github.com/ashviyer/Kg-algorithms.git"
GIT_EMAIL="aishwaryaiyer18@example.com"
GIT_NAME="ashviyer"
# ─────────────────────────────────────────────────────────────────────────────

git config --global user.email "$GIT_EMAIL"
git config --global user.name "$GIT_NAME"

# Create .gitignore
cat << 'EOF' > .gitignore
# Large binary/output files
*.pkl
embedding_file*.pkl

# Generated output files (can be recreated)
graph.txt
nodetypes.tsv
dis_sim.tsv
similarity_graph.txt
similarty_graph.txt
similarty_graph_drugs.tsv
hierarchy.csv
preprocessed_graph.csv
results1_04.csv
_tmp_query_pairs.tsv

# Folders with generated outputs and old workflows
dda_files/
results/
venv/
0.7 workflow/
0.9 workflow/

# OS
.DS_Store
__pycache__/
*.pyc
EOF

git init
git remote add origin "$REPO_URL"
git add .
git commit -m "Initial commit - Dreamwalk AOP workflow"
git branch -M main
git push -u origin main
