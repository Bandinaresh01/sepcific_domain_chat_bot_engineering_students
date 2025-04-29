#!/bin/bash

# =============================================
# Initialize Git and set up repository
# =============================================

# Remove existing Git history (if any)
rm -rf .git

# Initialize new Git repo
git init

# =============================================
# Create .gitignore file to exclude large files
# =============================================
cat > .gitignore << 'EOL'
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
*.egg
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.manifest
*.spec

# Virtual environment
venv/
env/
ENV/
.env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data files
*.pdf
*.faiss
*.pkl
*.pickle
*.h5
*.hdf5
*.pt
*.bin
*.dat

# Large files
*.dll
*.lib
*.pth
*.zip
*.tar.gz

# Logs and databases
*.log
*.sql
*.sqlite

# Environment files
.env
.env.local
.env.development
.env.test
.env.production

# FAISS specific
*.index
*.faissindex
EOL

# =============================================
# Add files and commit
# =============================================

# Stage all files (except those in .gitignore)
git add .

# Commit changes
git commit -m "Initial project commit"

# =============================================
# Set up remote and push
# =============================================

# Add remote repository
git remote add origin https://github.com/Bandinaresh01/sepcific_domain_chat_bot_engineering_students.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub (handle any conflicts)
git pull origin main --allow-unrelated-histories || true
git push -u origin main