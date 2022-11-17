#!/usr/bin/env bash

# ln -s ./oneformer/modeling/pixel_decoder/ops/ ./
# ls
# cd ops/ && bash make.sh && cd ..
echo '----------------------------------------------------------------'
echo '----------------------------------------------------------------'
pip3 freeze | grep MultiScaleDeformableAttention
pip3 freeze | grep torch
pip3 freeze | grep detectron2
pip3 freeze | grep natten
echo '----------------------------------------------------------------'
echo '----------------------------------------------------------------'

# echo '----------------------------------------------------------------'
# echo '----------------------------------------------------------------'
# cd /home/user/.pyenv/versions/3.8.15/lib/python3.8/site-packages
# ls
# ls | grep MultiScale
# echo '----------------------------------------------------------------'
# echo '----------------------------------------------------------------'
