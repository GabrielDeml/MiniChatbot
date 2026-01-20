#!/usr/bin/env python3
"""Gradio Chatbot Interface - Optimized for size and efficiency."""

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from train_chatbot import Transformer, Config

try:
    import spaces
    ZERO_GPU_AVAILABLE = True
except ImportError:
    ZERO_GPU_AVAILABLE = False

MODEL, TOKENIZER, DEVICE = None, None, None


def load_model():
    """Load the trained model and tokenizer."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'})
    
    config = Config()
    model = Transformer(
        vocab_size=len(tokenizer),
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        d_ff=config.D_FF,
        max_len=config.MAX_LEN,
        dropout=0.0,
        pad_idx=tokenizer.pad_token_id
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device, weights_only=True))
        print(f"Loaded model from {config.MODEL_PATH}")
    except FileNotFoundError:
        print(f"Warning: No trained model found at {config.MODEL_PATH}")
    
    model.eval()
    return model, tokenizer, device


def generate_response(model, tokenizer, device, input_text, temperature=0.8, max_length=50):
    """Generate a response with KV-cache for efficient autoregressive decoding."""
    if not input_text.strip():
        return ""
    
    pad_id, bos_id, eos_id = tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id
    
    with torch.inference_mode():
        input_tokens = [bos_id] + tokenizer.encode(input_text, add_special_tokens=False) + [eos_id]
        src = torch.tensor([input_tokens], device=device)
        src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
        enc_out = model.encode(src, src_mask)
        
        cur_tok = torch.tensor([[bos_id]], device=device)
        generated, kv_cache = [], None
        
        for _ in range(max_length):
            dec_out, kv_cache = model.decode(cur_tok, enc_out, src_mask, None, kv_cache)
            logits = model.output_projection(dec_out[:, -1, :])
            
            if temperature <= 0.1:
                next_token = logits.argmax(dim=-1).item()
            else:
                next_token = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).item()
            
            if next_token == eos_id:
                break
            
            generated.append(next_token)
            cur_tok = torch.tensor([[next_token]], device=device)
        
        return tokenizer.decode(generated, skip_special_tokens=True)


def get_model():
    """Lazy load the model."""
    global MODEL, TOKENIZER, DEVICE
    if MODEL is None:
        MODEL, TOKENIZER, DEVICE = load_model()
        print(f"Model loaded on {DEVICE}")
    return MODEL, TOKENIZER, DEVICE


def _chat_impl(message, history, temperature, max_length):
    model, tokenizer, device = get_model()
    return generate_response(model, tokenizer, device, message, temperature, int(max_length))


if ZERO_GPU_AVAILABLE:
    @spaces.GPU
    def chat(message, history, temperature, max_length):
        return _chat_impl(message, history, temperature, max_length)
else:
    def chat(message, history, temperature, max_length):
        return _chat_impl(message, history, temperature, max_length)


with gr.Blocks(title="Mini Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Mini Chatbot
    An ultra-compact GPT-style chatbot (~1M parameters) optimized for efficiency.
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="Conversation", height=400, show_copy_button=True, type="tuples")
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            temperature = gr.Slider(0.1, 2.0, 0.8, step=0.1, label="Temperature")
            max_length = gr.Slider(10, 100, 50, step=5, label="Max Length")
    
    def respond(message, chat_history, temp, max_len):
        if not message.strip():
            return chat_history, ""
        chat_history.append((message, chat(message, chat_history, temp, max_len)))
        return chat_history, ""
    
    msg.submit(respond, [msg, chatbot, temperature, max_length], [chatbot, msg])
    submit_btn.click(respond, [msg, chatbot, temperature, max_length], [chatbot, msg])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch()
