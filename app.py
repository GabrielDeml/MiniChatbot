#!/usr/bin/env python3
"""
Gradio Chatbot Interface for Hugging Face Spaces

Deploy to Hugging Face Spaces by uploading:
- app.py (this file)
- train_chatbot.py (model definitions)
- best_chatbot_model.pt (trained weights)
- requirements.txt
"""

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

# Import model components from training script
from train_chatbot import Transformer, Config


def load_model():
    """Load the trained model and tokenizer."""
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Setup tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {
        'pad_token': '<PAD>',
        'bos_token': '<BOS>',
        'eos_token': '<EOS>',
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Create model
    config = Config()
    vocab_size = len(tokenizer)
    
    model = Transformer(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        d_ff=config.D_FF,
        max_len=config.MAX_LEN,
        dropout=config.DROPOUT,
        pad_idx=tokenizer.pad_token_id
    ).to(device)
    
    # Load trained weights
    try:
        model.load_state_dict(
            torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
        )
        print(f"Loaded model from {config.MODEL_PATH}")
    except FileNotFoundError:
        print(f"Warning: No trained model found at {config.MODEL_PATH}")
        print("The model will generate random responses until trained.")
    
    model.eval()
    return model, tokenizer, device


def generate_response(model, tokenizer, device, input_text, temperature=0.8, max_length=50):
    """Generate a response to the input text."""
    if not input_text.strip():
        return ""
    
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        # Encode input
        input_tokens = tokenizer.encode(input_text, add_special_tokens=False)
        input_tokens = [bos_id] + input_tokens + [eos_id]
        encoder_input = torch.tensor([input_tokens]).to(device)
        
        # Create source mask
        src_mask = (encoder_input != pad_id).unsqueeze(1).unsqueeze(2)
        encoder_output = model.encoder(encoder_input, src_mask)
        
        # Start decoding
        decoder_input = torch.tensor([[bos_id]]).to(device)
        generated_tokens = []
        
        for _ in range(max_length):
            tgt_len = decoder_input.size(1)
            tgt_pad_mask = (decoder_input != pad_id).unsqueeze(1).unsqueeze(2)
            causal_mask = model.generate_causal_mask(tgt_len, device)
            tgt_mask = tgt_pad_mask & (causal_mask == 0).unsqueeze(0)
            
            decoder_output = model.decoder(decoder_input, encoder_output, src_mask, tgt_mask)
            logits = model.output_projection(decoder_output[:, -1, :])
            
            # Sample with temperature
            if temperature <= 0.1:
                next_token = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == eos_id:
                break
            
            generated_tokens.append(next_token)
            decoder_input = torch.cat([
                decoder_input,
                torch.tensor([[next_token]]).to(device)
            ], dim=1)
        
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# Load model globally
print("Loading model...")
MODEL, TOKENIZER, DEVICE = load_model()
print(f"Model loaded on {DEVICE}")


def chat(message, history, temperature, max_length):
    """Chat function for Gradio interface."""
    response = generate_response(
        MODEL, TOKENIZER, DEVICE,
        message,
        temperature=temperature,
        max_length=int(max_length)
    )
    return response


def clear_chat():
    """Clear chat history."""
    return [], ""


# Create Gradio interface
with gr.Blocks(
    title="Transformer Chatbot",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown("""
    # Transformer Chatbot
    
    A conversational AI built from scratch using the Transformer architecture, 
    trained on the DailyDialog dataset.
    
    **Model Details:**
    - Architecture: Encoder-Decoder Transformer
    - Parameters: ~44M
    - Training: DailyDialog conversational dataset
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                show_copy_button=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Type a message and press Enter...",
                    scale=4,
                    show_label=False
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.8,
                step=0.1,
                label="Temperature",
                info="Higher = more creative, Lower = more focused"
            )
            
            max_length = gr.Slider(
                minimum=10,
                maximum=100,
                value=50,
                step=5,
                label="Max Response Length",
                info="Maximum tokens in response"
            )
            
            gr.Markdown("""
            ---
            ### Tips
            - Try casual greetings like "Hello!" or "How are you?"
            - Ask about daily topics
            - Lower temperature for more predictable responses
            """)
    
    # Event handlers
    def respond(message, chat_history, temp, max_len):
        if not message.strip():
            return chat_history, ""
        bot_response = chat(message, chat_history, temp, max_len)
        chat_history.append((message, bot_response))
        return chat_history, ""
    
    msg.submit(respond, [msg, chatbot, temperature, max_length], [chatbot, msg])
    submit_btn.click(respond, [msg, chatbot, temperature, max_length], [chatbot, msg])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])


if __name__ == "__main__":
    demo.launch()
