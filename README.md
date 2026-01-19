---
title: Transformer Chatbot
emoji: 💬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Transformer Chatbot

A conversational AI built from scratch using the Transformer architecture, trained on the DailyDialog dataset.

## Model Details

- **Architecture:** Encoder-Decoder Transformer
- **Parameters:** ~44M
- **Embedding Dimension:** 256
- **Attention Heads:** 8
- **Layers:** 4 (encoder) + 4 (decoder)
- **Training Data:** DailyDialog conversational dataset

## Usage

Type a message in the chat box and press Enter or click Send. Adjust the temperature slider to control response creativity:
- Lower temperature (0.1-0.5): More focused, predictable responses
- Higher temperature (0.8-1.5): More creative, varied responses

## Files

- `app.py` - Gradio interface
- `train_chatbot.py` - Model architecture and training code
- `best_chatbot_model.pt` - Trained model weights
- `requirements.txt` - Python dependencies

## Training

To train your own model:

```bash
python train_chatbot.py --epochs 10
```

## License

MIT
