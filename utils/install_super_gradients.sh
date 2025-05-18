#!/bin/bash

echo ""
echo "######################################################################################"
echo "###                                                                                ###"
echo "###   ALERT: It is highly recommended to install SuperGradients in a Python        ###"
echo "###          virtual environment (venv, conda, etc.).                            ###"
echo "###                                                                                ###"
echo "###   Using a virtual environment prevents conflicts with your system Python       ###"
echo "###   and other projects.                                                          ###"
echo "###                                                                                ###"
echo "###   PLEASE ACTIVATE YOUR VIRTUAL ENVIRONMENT *BEFORE* RUNNING THIS SCRIPT.     ###"
echo "###                                                                                ###"
echo "###   Example (venv):                                                              ###"
echo "###     python3 -m venv my_sg_venv                                                 ###"
echo "###     source my_sg_venv/bin/activate                                             ###"
echo "###                                                                                ###"
echo "###   Example (conda):                                                               ###"
echo "###     conda create -n my_sg_env python=3.x  # Use your desired Python version   ###"
echo "###     conda activate my_sg_env                                                   ###"
echo "###                                                                                ###"
echo "###   If you are already in a virtual environment, you can ignore this message.    ###"
echo "###                                                                                ###"
echo "######################################################################################"
echo ""

read -p "Do you wish to proceed? (y/N): " confirm
confirm=${confirm,,} # Convert input to lowercase

if [[ "$confirm" != "y" ]]; then
    echo "Installation cancelled by user."
    exit 0 # Exit cleanly if the user doesn't confirm
fi

echo "Proceeding with installation..."
echo ""

# pull repo
git clone https://github.com/Deci-AI/super-gradients.git
cd super-gradients

# install super-gradients to a specific PR that fixes URLS in the code for model weights, see https://github.com/Deci-AI/super-gradients/pull/2061
git fetch origin pull/2061/head:url_fixes
git switch url_fixes
pip3 install -r requirements.txt && python3 -m pip install -e .

# To install openvino, required for exporting models to openvino model optimizer format (.bin, .xml)
python3 -m pip install openvino
