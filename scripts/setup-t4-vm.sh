#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Provision the T4 test VM with NVIDIA drivers, Docker, and the GenieSim image.
#
# Run this ONCE after creating the VM with create-t4-test-vm.sh.
# Runs FROM your Mac — SSHes into the VM to execute each step.
#
# Usage:
#   bash scripts/setup-t4-vm.sh
#
# Duration: ~30-40 minutes (mostly Docker image build).
# =============================================================================

VM_NAME="${T4_VM_NAME:-geniesim-t4-test}"
VM_ZONE="${T4_VM_ZONE:-us-east1-b}"
REPO_BRANCH="${REPO_BRANCH:-main}"

# GitHub repo URL (used for cloning on the VM)
REPO_URL="${REPO_URL:-https://github.com/$(git remote get-url origin 2>/dev/null | sed 's|.*github.com[:/]||;s|\.git$||' || echo 'your-org/BlueprintPipeline')}"

ssh_cmd() {
  gcloud compute ssh "${VM_NAME}" --zone="${VM_ZONE}" -- "$@"
}

echo "============================================================"
echo " Setting up T4 VM: ${VM_NAME}"
echo "============================================================"

# ── Step 0: Wait for SSH ─────────────────────────────────────────────────────
echo "[setup-t4] Waiting for SSH access..."
for i in $(seq 1 30); do
  if ssh_cmd "echo 'SSH ready'" 2>/dev/null; then
    break
  fi
  echo "  Attempt $i/30..."
  sleep 10
done

# ── Step 1: Install NVIDIA driver ────────────────────────────────────────────
echo "[setup-t4] Step 1/6: Installing NVIDIA driver..."
ssh_cmd bash -c "'
  if nvidia-smi &>/dev/null; then
    echo \"NVIDIA driver already installed:\"
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
  else
    echo \"Installing NVIDIA driver...\"
    sudo apt-get update -qq
    sudo apt-get install -y -qq linux-headers-\$(uname -r)
    # Use the NVIDIA driver PPA for latest stable
    sudo apt-get install -y -qq software-properties-common
    sudo add-apt-repository -y ppa:graphics-drivers/ppa
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-driver-550
    echo \"Driver installed. Rebooting...\"
    sudo reboot
  fi
'"

# Wait for reboot if it happened
echo "[setup-t4] Waiting for VM to come back after potential reboot..."
sleep 30
for i in $(seq 1 30); do
  if ssh_cmd "nvidia-smi &>/dev/null && echo 'GPU ready'" 2>/dev/null; then
    break
  fi
  echo "  Waiting for GPU... (attempt $i/30)"
  sleep 10
done

echo "[setup-t4] GPU status:"
ssh_cmd nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# ── Step 2: Install Docker + NVIDIA Container Toolkit ────────────────────────
echo "[setup-t4] Step 2/6: Installing Docker + nvidia-container-toolkit..."
ssh_cmd bash -c "'
  if docker --version &>/dev/null && nvidia-container-cli --version &>/dev/null; then
    echo \"Docker and nvidia-container-toolkit already installed.\"
  else
    # Docker
    if ! docker --version &>/dev/null; then
      curl -fsSL https://get.docker.com | sudo sh
      sudo usermod -aG docker \$USER
    fi

    # NVIDIA Container Toolkit
    if ! nvidia-container-cli --version &>/dev/null; then
      distribution=\$(. /etc/os-release; echo \$ID\$VERSION_ID)
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -s -L \"https://nvidia.github.io/libnvidia-container/\${distribution}/libnvidia-container.list\" | \
        sed \"s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g\" | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
      sudo apt-get update -qq
      sudo apt-get install -y -qq nvidia-container-toolkit
      sudo nvidia-ctk runtime configure --runtime=docker
      sudo systemctl restart docker
    fi

    echo \"Docker and nvidia-container-toolkit installed.\"
  fi
'"

# ── Step 3: Install Xorg for headless display ────────────────────────────────
echo "[setup-t4] Step 3/6: Installing Xorg..."
ssh_cmd bash -c "'
  if command -v Xorg &>/dev/null; then
    echo \"Xorg already installed.\"
  else
    sudo apt-get update -qq
    sudo apt-get install -y -qq xserver-xorg-core xserver-xorg-video-nvidia
    echo \"Xorg installed.\"
  fi
'"

# ── Step 4: Clone BlueprintPipeline repo ─────────────────────────────────────
echo "[setup-t4] Step 4/6: Cloning BlueprintPipeline..."
ssh_cmd bash -c "'
  if [ -d ~/BlueprintPipeline/.git ]; then
    echo \"Repo already exists. Pulling latest...\"
    cd ~/BlueprintPipeline && git pull origin ${REPO_BRANCH} || true
  else
    git clone --branch ${REPO_BRANCH} ${REPO_URL} ~/BlueprintPipeline
  fi
  echo \"Repo ready at ~/BlueprintPipeline\"
'"

# ── Step 5: NGC login for Isaac Sim base image ───────────────────────────────
echo "[setup-t4] Step 5/6: NGC authentication..."
echo "[setup-t4] NOTE: The Isaac Sim Docker image requires NGC authentication."
echo "[setup-t4] If the build fails with 'unauthorized', you need to run on the VM:"
echo "           sudo docker login nvcr.io -u '\$oauthtoken' -p '<your-ngc-api-key>'"
echo ""

# ── Step 6: Build Docker image ───────────────────────────────────────────────
echo "[setup-t4] Step 6/6: Building geniesim-server Docker image..."
echo "[setup-t4] This takes ~20-30 minutes (Isaac Sim base image is large)."
ssh_cmd bash -c "'
  cd ~/BlueprintPipeline

  # Check if image already exists
  if sudo docker images geniesim-server:latest --format \"{{.ID}}\" | head -1 | grep -q .; then
    echo \"geniesim-server:latest image already exists.\"
    sudo docker images geniesim-server:latest --format \"table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}\"
    echo \"To rebuild: sudo docker build -f Dockerfile.geniesim-server-nocurobo -t geniesim-server:latest .\"
  else
    echo \"Building geniesim-server image...\"
    sudo docker build -f Dockerfile.geniesim-server-nocurobo -t geniesim-server:latest . 2>&1 | tail -20
    echo \"Build complete.\"
    sudo docker images geniesim-server:latest --format \"table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}\"
  fi
'"

echo ""
echo "============================================================"
echo " T4 VM Setup Complete"
echo "============================================================"
echo "VM Name: ${VM_NAME}"
echo "Zone:    ${VM_ZONE}"
echo ""
echo "Next: bash scripts/run-t4-rgb-test.sh"
echo ""
echo "To stop (save money): gcloud compute instances stop ${VM_NAME} --zone=${VM_ZONE}"
echo "============================================================"
