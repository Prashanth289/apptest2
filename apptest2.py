import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image

import numpy as np
import os
import time
# import spaces  # Import spaces for ZeroGPU compatibility


# Load model and processor
model_path = r"C:\Users\Mourya C E\Desktop\jjanus\Janus\models\Janus-Pro\Janus-Pro-1B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                             language_config=language_config,
                                             trust_remote_code=True)
if torch.cuda.is_available():
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
else:
    vl_gpt = vl_gpt.to(torch.float16)

vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.inference_mode()
# @spaces.GPU(duration=120) 
# Multimodal Understanding function
def multimodal_understanding(image, question, seed, top_p, temperature):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
    
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer


def generate(input_ids,
             width,
             height,
             temperature: float = 1,
             parallel_size: int = 5,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16):
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(cuda_device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(cuda_device)

    pkv = None
    for i in range(image_token_num_per_image):
        with torch.no_grad():
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds,
                                                use_cache=True,
                                                past_key_values=pkv)
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    

    patches = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                 shape=[parallel_size, 8, width // patch_size, height // patch_size])

    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img



@torch.inference_mode()
# @spaces.GPU(duration=120)  # Specify a duration to avoid timeout
def generate_image(prompt,
                   seed=None,
                   guidance=5,
                   t2i_temperature=1.0,
                   num_images=5):
    # Clear CUDA cache and avoid tracking gradients
    torch.cuda.empty_cache()
    # Set the seed for reproducible results
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    width = 384
    height = 384
    parallel_size = num_images
    
    with torch.no_grad():
        messages = [{'role': '<|User|>', 'content': prompt},
                    {'role': '<|Assistant|>', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=messages,
                                                                   sft_format=vl_chat_processor.sft_format,
                                                                   system_prompt='')
        text = text + vl_chat_processor.image_start_tag
        
        input_ids = torch.LongTensor(tokenizer.encode(text))
        output, patches = generate(input_ids,
                                   width // 16 * 16,
                                   height // 16 * 16,
                                   cfg_weight=guidance,
                                   parallel_size=parallel_size,
                                   temperature=t2i_temperature)
        images = unpack(patches,
                        width // 16 * 16,
                        height // 16 * 16,
                        parallel_size=parallel_size)

        return [Image.fromarray(images[i]).resize((768, 768), Image.LANCZOS) for i in range(parallel_size)]
        

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="teal",
    neutral_hue="slate",
    font=["sans-serif", "system-ui"],
    font_mono=["monospace", "system-ui"],
    radius_size=gr.themes.sizes.radius_lg,
    text_size=gr.themes.sizes.text_lg,
), css="""
.gallery-container {
    max-height: 600px;
    overflow-y: auto;
    padding: 10px;
    background: linear-gradient(135deg, #1a3a4a, #2a4a5a);
    border-radius: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.gallery-container::-webkit-scrollbar {
    width: 10px;
}

.gallery-container::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

.gallery-container::-webkit-scrollbar-thumb {
    background: rgba(32, 178, 170, 0.5);
    border-radius: 10px;
}

.gallery-container::-webkit-scrollbar-thumb:hover {
    background: rgba(32, 178, 170, 0.7);
}

.gradio-container {
    background: linear-gradient(135deg, #1a3a4a, #2a4a5a, #3a5a6a);
    min-height: 100vh;
}

.gradio-container .panel {
    background: rgba(26, 58, 74, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(32, 178, 170, 0.2);
}

.gradio-container .panel:hover {
    background: rgba(26, 58, 74, 0.8);
    border: 1px solid rgba(32, 178, 170, 0.4);
}

.gradio-container button {
    background: linear-gradient(45deg, #20B2AA, #008B8B);
    border: none;
    color: white;
    border-radius: 15px;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.gradio-container button:hover {
    background: linear-gradient(45deg, #008B8B, #20B2AA);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

.gradio-container .slider {
    background: linear-gradient(90deg, #20B2AA, #008B8B);
    border-radius: 10px;
    height: 8px;
}

.gradio-container input[type="text"], 
.gradio-container input[type="number"] {
    background: rgba(26, 58, 74, 0.7);
    border: 1px solid rgba(32, 178, 170, 0.3);
    color: white;
    border-radius: 15px;
    padding: 10px 15px;
    transition: all 0.3s ease;
}

.gradio-container input[type="text"]:focus, 
.gradio-container input[type="number"]:focus {
    background: rgba(26, 58, 74, 0.9);
    border: 1px solid rgba(32, 178, 170, 0.6);
    box-shadow: 0 0 10px rgba(32, 178, 170, 0.3);
}

.gradio-container .markdown {
    background: rgba(26, 58, 74, 0.7);
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid rgba(32, 178, 170, 0.2);
}

.gradio-container .markdown h1,
.gradio-container .markdown h2,
.gradio-container .markdown h3 {
    color: #20B2AA;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.gradio-container .label {
    color: #20B2AA;
    font-weight: 600;
    margin-bottom: 8px;
}

.gradio-container .output {
    background: rgba(26, 58, 74, 0.7);
    border-radius: 20px;
    padding: 20px;
    border: 1px solid rgba(32, 178, 170, 0.2);
}
""") as demo:
    gr.Markdown(value="# Multimodal Understanding")
    with gr.Row():
        image_input = gr.Image()
        with gr.Column():
            question_input = gr.Textbox(label="Question", placeholder="Ask any question about the image...")
            und_seed_input = gr.Number(label="Seed", precision=0, value=42, info="Random seed for reproducibility")
            top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="Top P", info="Controls diversity of text generation")
            temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="Temperature", info="Controls randomness in text generation")
        
    understanding_button = gr.Button("Chat", variant="primary")
    understanding_output = gr.Textbox(label="Response")
    
    gr.Markdown(value="# Text-to-Image Generation")

    with gr.Row():
        with gr.Column():
            cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight", info="Higher values make the image more closely follow the prompt")
            t2i_temperature = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.05, label="Temperature", info="Controls randomness in image generation")
            num_images = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Images", info="How many variations to generate")

    prompt_input = gr.Textbox(label="Prompt", placeholder="Enter a detailed description of the image you want to generate...")
    seed_input = gr.Number(label="Seed (Optional)", precision=0, value=12345, info="Random seed for reproducibility")

    generation_button = gr.Button("Generate Images", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            image_output = gr.Gallery(
                label="Generated Images",
                columns=2,
                rows=2,
                height=600,
                show_label=True,
                object_fit="contain",
                preview=True,
                allow_preview=True,
                show_download_button=True,
                elem_id="gallery-container",
                elem_classes=["scrollable-gallery"]
            )
    
    understanding_button.click(
        multimodal_understanding,
        inputs=[image_input, question_input, und_seed_input, top_p, temperature],
        outputs=understanding_output
    )
    
    generation_button.click(
        fn=generate_image,
        inputs=[prompt_input, seed_input, cfg_weight_input, t2i_temperature, num_images],
        outputs=image_output
    )

demo.launch(share=True)
# demo.queue(concurrency_count=1, max_size=10).launch(server_name="0.0.0.0", server_port=37906, root_path="/path")
