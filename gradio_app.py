import torch

print("Installed the dependencies!")

import numpy as np
from PIL import Image
import cv2
import imutils

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
)

from demo.defaults import DefaultPredictor
from demo.visualizer import Visualizer, ColorMode

import gradio as gr
from huggingface_hub import hf_hub_download

KEY_DICT = {"Cityscapes (19 classes)": "cityscapes",
            "COCO (133 classes)": "coco",
            "ADE20K (150 classes)": "ade20k",}

SWIN_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
            "coco": "configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",}

SWIN_MODEL_DICT = {"cityscapes": hf_hub_download(repo_id="shi-labs/swin_l_oneformer_cityscapes", 
                                            filename="250_16_swin_l_oneformer_cityscapes_90k.pth"),
              "coco": hf_hub_download(repo_id="shi-labs/swin_l_oneformer_coco", 
                                            filename="150_16_swin_l_oneformer_coco_100ep.pth"),
              "ade20k": hf_hub_download(repo_id="shi-labs/swin_l_oneformer_ade20k", 
                                            filename="250_16_swin_l_oneformer_ade20k_160k.pth")
            }

DINAT_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_dinat_large_bs16_90k.yaml",
            "coco": "configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml",}

DINAT_MODEL_DICT = {"cityscapes": hf_hub_download(repo_id="shi-labs/dinat_l_oneformer_cityscapes", 
                                            filename="250_16_dinat_l_oneformer_cityscapes_90k.pth"),
              "coco": hf_hub_download(repo_id="shi-labs/dinat_l_oneformer_coco", 
                                            filename="150_16_dinat_l_oneformer_coco_100ep.pth"),
              "ade20k": hf_hub_download(repo_id="shi-labs/dinat_l_oneformer_ade20k", 
                                            filename="250_16_dinat_l_oneformer_ade20k_160k.pth")
            }

MODEL_DICT = {"DiNAT-L": DINAT_MODEL_DICT,
        "Swin-L": SWIN_MODEL_DICT }

CFG_DICT = {"DiNAT-L": DINAT_CFG_DICT,
        "Swin-L": SWIN_CFG_DICT }

WIDTH_DICT = {"cityscapes": 512,
              "coco": 512,
              "ade20k": 640}

cpu_device = torch.device("cpu")

PREDICTORS = {
    "DiNAT-L": {
        "Cityscapes (19 classes)": None,
        "COCO (133 classes)": None,
        "ADE20K (150 classes)": None
    },
    "Swin-L": {
        "Cityscapes (19 classes)": None,
        "COCO (133 classes)": None,
        "ADE20K (150 classes)": None
    }
}

def setup_predictors():
    for dataset in ["Cityscapes (19 classes)", "COCO (133 classes)", "ADE20K (150 classes)"]:
        for backbone in ["DiNAT-L", "Swin-L"]:
            cfg = setup_cfg(dataset, backbone)
            PREDICTORS[backbone][dataset] = DefaultPredictor(cfg)

def setup_cfg(dataset, backbone):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_oneformer_config(cfg)
    add_dinat_config(cfg)
    dataset = KEY_DICT[dataset]
    cfg_path = CFG_DICT[backbone][dataset]
    cfg.merge_from_file(cfg_path)
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cuda'
    else:
        cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = MODEL_DICT[backbone][dataset]
    cfg.freeze()
    return cfg

def setup_modules(dataset, backbone):
    cfg = setup_cfg(dataset, backbone)
    # predictor = DefaultPredictor(cfg)
    predictor = PREDICTORS[backbone][dataset]
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
    )
    if 'cityscapes_fine_sem_seg_val' in cfg.DATASETS.TEST_PANOPTIC[0]:
        from cityscapesscripts.helpers.labels import labels
        stuff_colors = [k.color for k in labels if k.trainId != 255]
        metadata = metadata.set(stuff_colors=stuff_colors)
    
    return predictor, metadata

def panoptic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    out = visualizer.draw_panoptic_seg_predictions(
        panoptic_seg.to(cpu_device), segments_info, alpha=0.5
    )
    visualizer_map = Visualizer(img[:, :, ::-1], is_img=False, metadata=metadata, instance_mode=ColorMode.IMAGE)
    out_map = visualizer_map.draw_panoptic_seg_predictions(
        panoptic_seg.to(cpu_device), segments_info, alpha=1, is_text=False
    )
    return out, out_map

def instance_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "instance")
    instances = predictions["instances"].to(cpu_device)
    out = visualizer.draw_instance_predictions(predictions=instances, alpha=0.5)
    visualizer_map = Visualizer(img[:, :, ::-1], is_img=False, metadata=metadata, instance_mode=ColorMode.IMAGE)
    out_map = visualizer_map.draw_instance_predictions(predictions=instances, alpha=1, is_text=False)
    return out, out_map

def semantic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "semantic")
    out = visualizer.draw_sem_seg(
        predictions["sem_seg"].argmax(dim=0).to(cpu_device), alpha=0.5
    )
    visualizer_map = Visualizer(img[:, :, ::-1], is_img=False, metadata=metadata, instance_mode=ColorMode.IMAGE)
    out_map = visualizer_map.draw_sem_seg(
        predictions["sem_seg"].argmax(dim=0).to(cpu_device), alpha=1, is_text=False
    )
    return out, out_map

TASK_INFER = {"the task is panoptic": panoptic_run, "the task is instance": instance_run, "the task is semantic": semantic_run}

def segment(path, task, dataset, backbone):
    predictor, metadata = setup_modules(dataset, backbone)
    img = cv2.imread(path)
    width = WIDTH_DICT[KEY_DICT[dataset]]
    img = imutils.resize(img, width=width)
    out, out_map = TASK_INFER[task](img, predictor, metadata)
    out = Image.fromarray(out.get_image())
    out_map = Image.fromarray(out_map.get_image())
    return out, out_map

title = "OneFormer: One Transformer to Rule Universal Image Segmentation"

description = "<p style='color: #E0B941; font-size: 16px; font-weight: w600; text-align: center'> <a style='color: #E0B941;' href='https://praeclarumjj3.github.io/oneformer/' target='_blank'>Project Page</a> | <a style='color: #E0B941;' href='https://arxiv.org/abs/2211.06220' target='_blank'>OneFormer: One Transformer to Rule Universal Image Segmentation</a> | <a style='color: #E0B941;' href='https://github.com/SHI-Labs/OneFormer' target='_blank'>Github</a></p>" \
            + "<p style='color:royalblue; margin: 10px; font-size: 16px; font-weight: w400;'>  \
                [Note: Inference on CPU may take upto 2 minutes.] This is the official gradio demo for our paper <span style='color:#E0B941;'>OneFormer: One Transformer to Rule Universal Image Segmentation</span> To use OneFormer: <br> \
                (1) <span style='color:#E0B941;'>Upload an Image</span> or <span style='color:#E0B941;'> select a sample image from the examples</span> <br>  \
                (2) Select the value of the <span style='color:#E0B941;'>Task Token Input</span> <br>\
                (3) Select the <span style='color:#E0B941;'>Model</span> </p>"

# article = 

# css = ".image-preview {height: 32rem; width: auto;} .output-image {height: 32rem; width: auto;} .panel-buttons { display: flex; flex-direction: row;}"

setup_predictors()

gradio_inputs = [gr.Image(source="upload", tool=None, label="Input Image",type="filepath"),
            gr.inputs.Radio(choices=["the task is panoptic" ,"the task is instance", "the task is semantic"], type="value", default="the task is panoptic", label="Task Token Input"),
            gr.inputs.Radio(choices=["COCO (133 classes)" ,"Cityscapes (19 classes)", "ADE20K (150 classes)"], type="value", default="Cityscapes (19 classes)", label="Model"),
            gr.inputs.Radio(choices=["DiNAT-L" ,"Swin-L"], type="value", default="DiNAT-L", label="Backbone"),
            ]
gradio_outputs = [gr.Image(type="pil", label="Segmentation Overlay"), gr.Image(type="pil", label="Segmentation Map")]


examples = [["examples/coco.jpeg", "the task is panoptic", "COCO (133 classes)", "DiNAT-L"],
            ["examples/cityscapes.png", "the task is panoptic", "Cityscapes (19 classes)", "DiNAT-L"],
            ["examples/ade20k.jpeg", "the task is panoptic", "ADE20K (150 classes)", "DiNAT-L"]]


iface = gr.Interface(fn=segment, inputs=gradio_inputs,
                     outputs=gradio_outputs,
                     examples_per_page=5,
                     allow_flagging="never",
                     examples=examples, title=title,
                     description=description)

iface.launch(enable_queue=True)