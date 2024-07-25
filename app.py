#!/usr/bin/env python
import gradio as gr
import random
from lib.predict import predict, predict_v2, preview
# ---
# import numpy as np
# import matplotlib.pyplot as plt
# from io import BytesIO
# from PIL import Image

# def generate_image():
#     # Generate a random image using matplotlib
#     fig, ax = plt.subplots()
#     ax.imshow(np.random.rand(512, 512, 3))
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     plt.close(fig)
    
#     # Convert to PIL Image
#     img = Image.open(buf)
    
#     return [img]
# ---

def got_inputs(gallery_in):
    return gr.Button.update(interactive=True)

with gr.Blocks() as demo:
    gr.Markdown("# Bird Detector with Bounding Boxes")
    gr.Markdown("Upload your image or use generated images and then click **Predict** to see the output.")
    
    btn_gen_img = gr.Button("Generate images", scale=0, min_width=200)
    
    gallery_in = gr.Gallery(
        label="Chosen images",
        show_label=False,
        elem_id="gallery_in",
        columns=[5],
        rows=[2],
        object_fit="contain",
        height="auto",
        interactive=True,
        type="numpy",
        # value=preview()
    )

    btn_predict = gr.Button("Predict", scale=0, min_width=200)

    gallery_out = gr.Gallery(
        label="Predicted images",
        show_label=False,
        elem_id="gallery_out",
        columns=[5],
        rows=[2],
        object_fit="contain",
        height="auto",
        type="numpy",
    )
    # event listeners
    btn_gen_img.click(preview, None, gallery_in)
    btn_predict.click(predict_v2, gallery_in, gallery_out)

if __name__ == "__main__":
    demo.launch()

