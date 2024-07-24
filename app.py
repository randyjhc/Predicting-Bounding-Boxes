#!/usr/bin/env python
# %%
import gradio as gr
import random
from lib.predict import predict
# ---
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def generate_image():
    # Generate a random image using matplotlib
    fig, ax = plt.subplots()
    ax.imshow(np.random.rand(512, 512, 3))
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    # Convert to PIL Image
    img = Image.open(buf)
    
    return [img]
# ---

def fake_gan():
    images = [
        (random.choice(
            [
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-116b5e92936b766b7fdfc242649337f7.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1163530ca19b5cebe1b002b8ec67b6fc.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1116395d6e6a6581eef8b8038f4c8e55.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-11319be65db395d0e8e6855d18ddcef0.jpg",
            ]
        ), f"label {i}")
        for i in range(3)
    ]
    return images

with gr.Blocks() as demo:
    gr.Markdown("Upload your image or select a image below and then click **Predict** to see the output.")
    gallery = gr.Gallery(
        label="Chosen images",
        show_label=False,
        elem_id="gallery",
        columns=[3],
        rows=[1],
        object_fit="contain",
        height="auto",
        interactive=True
    )            
    
    btn = gr.Button("Generate images", scale=0)

    # btn.click(fake_gan, None, gallery)
    btn.click(predict, None, gallery)

if __name__ == "__main__":
    demo.launch()

