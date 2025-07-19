# 🧠 Stable SAM Diffusion Model

This project combines Meta's **Segment Anything Model (SAM)**, **CLIP embeddings**, and **Stable Diffusion Inpainting** to enable precise and intelligent image editing using natural language prompts.

> **Use case**: Click on an object → Mask it using SAM → Enter a text prompt → Modify the object using Stable Diffusion inpainting guided by your text!

---

## 🚀 Features

- 🖼️ **Image Upload** or Webcam Capture
- 🖱️ **Interactive Segmentation** using SAM (Segment Anything)
- ✏️ **Text Prompt-Based Editing** using Stable Diffusion
- 🤖 **CLIP + Diffusion** for semantic alignment
- 💻 **Gradio Interface** for easy interaction

---

## 🧩 Tech Stack

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [Stable Diffusion Inpainting](https://github.com/huggingface/diffusers)
- [CLIP](https://github.com/openai/CLIP)
- [Gradio](https://www.gradio.app/) for the UI
