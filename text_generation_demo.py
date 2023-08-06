import argparse
import torch

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer, model = None, None


def init_model(args):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, truncation_side="left", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.tokenizer_path, trust_remote_code=True).half().cuda()
    model = model.eval()


def batch_call(texts, skip_special_tokens=True, **kwargs):
    tokenized = tokenizer(texts, padding=True, return_tensors="pt")
    inputs = {key: value.cuda() for key, value in tokenized.items() if key != 'token_type_ids'}
    generate_ids = model.generate(**inputs, **kwargs)

    output =[]
    for tok, gen in zip(tokenized.input_ids, generate_ids):
        generated = tokenizer.decode(gen[len(tok):], skip_special_tokens=skip_special_tokens)
        output.append(generated)
    return output


def text_generation(texts, max_new_tokens, temperature, top_k, top_p):
    output = batch_call(texts, max_new_tokens=max_new_tokens, do_sample=True, top_k=top_k, top_p=top_p, temperature=temperature, eos_token_id=tokenizer.eos_token_id)
    return output[0]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=20014,
                        help="server port")
    parser.add_argument("--model_path", type=str, default="./model",
                        help="Path to the model. Specifies the file path to the pre-trained model to be used for text generation.")
    parser.add_argument("--tokenizer_path", type=str, default="./model",
                        help="Path to the tokenizer.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # initialize model and tokenizer
    init_model(args)

    with gr.Blocks() as demo:
        gr.Markdown(
            "# <center>{}</center>".format("XVERSE-13B Text Generation"))
        with gr.Row():
            with gr.Column():
                inputs = gr.inputs.Textbox(
                    lines=5, label="Input Text")  # input
                with gr.Column():
                    max_new_tokens = gr.Slider(maximum=512, value=100, minimum=1, step=1,
                                               label="max_new_tokens", interactive=True)  # max_new_tokens
                    temperature = gr.Slider(maximum=1.0, value=1.0, minimum=0.0, step=0.05,
                                            label='temperature', interactive=True)  # temperature
                    top_k = gr.Slider(maximum=50, value=50, minimum=0, step=1,
                                      label='Top K', interactive=True)  # top_k
                    top_p = gr.Slider(maximum=1, value=0.92, minimum=0,
                                      step=0.02, label='Top P', interactive=True)  # top_p

            with gr.Row():
                outputs = gr.inputs.Textbox(lines=2, label="Output Text")

        with gr.Row():
            submit_btn = gr.Button(value="生成", variant="secondary")
            reset_btn = gr.ClearButton(components=[inputs, outputs], value="清除", variant="secondary")

        submit_btn.click(fn=text_generation,
                         inputs=[inputs, max_new_tokens,
                                 temperature, top_k, top_p],
                         outputs=outputs)

    demo.launch(server_name="0.0.0.0", server_port=args.port)