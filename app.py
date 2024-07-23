#!/usr/bin/env python
# %%
import gradio as gr
from lib.predict import predict

def fake_gan():
    a = Image.new("RGB", (512, 512), "red")
    b = Image.new("RGB", (512, 512), "green")
    c = Image.new("RGB", (512, 512), "blue")
    images = [
        (random.choice(
            [
                a, b, c,
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
    gallery = gr.Gallery(
        label="Generated images",
        show_label=False,
        elem_id="gallery",
        columns=[3],
        rows=[1],
        object_fit="contain",
        height="auto"
    )
    btn = gr.Button("Generate images", scale=0)

    btn.click(fake_gan, None, gallery)


if __name__ == "__main__":
    demo.launch()

