import torch
import torch.nn.functional as F
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- 1. IMPORT YOUR MODEL HERE ---
# from your_module import SwinMoETex, SparseMoEFFN
# from your_tokenizer import Tokenizer

# Placeholder variables for initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None 
TOP_K = 2 # Your MoE top_k
NUM_EXPERTS = 8 # Your MoE num_experts

# Global dictionary to store intercepted routing data
# Format: { layer_name: [ array(batch*seq, top_k), array(batch*1, top_k), ... ] }
routing_history = {}

def get_router_hook(layer_name):
    """
    Creates a PyTorch hook to intercept the outputs of the router.
    """
    def hook(module, input, output):
        # output is the logits from self.router: shape (batch*seq_len, num_experts)
        logits = output.detach()
        gate_probs = F.softmax(logits, dim=-1)
        _, selected_experts = torch.topk(gate_probs, TOP_K, dim=-1)
        
        if layer_name not in routing_history:
            routing_history[layer_name] = []
        
        # Save the selected expert IDs as a numpy array
        routing_history[layer_name].append(selected_experts.cpu().numpy())
    return hook

def attach_hooks(model):
    """Finds all SparseMoEFFN routers and attaches hooks to them."""
    hooks = []
    layer_count = 0
    # Iterate through all submodules
    for name, module in model.named_modules():
        # Look specifically for the MoE class name (adjust if your class is named differently)
        if module.__class__.__name__ == "SparseMoEFFN":
            layer_name = f"MoE_Layer_{layer_count}"
            # Attach hook to the router linear layer inside the MoEFFN
            h = module.router.register_forward_hook(get_router_hook(layer_name))
            hooks.append(h)
            layer_count += 1
    return hooks

def generate_and_visualize(image, prompt, max_new_tokens, temperature):
    global routing_history
    routing_history.clear() # Reset history for the new run
    
    # --- 2. PREPROCESS INPUTS ---
    # Convert prompt to tokens (implement your actual tokenizer logic here)
    # start_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Mocking tokens & image for the sake of the standalone script:
    start_tokens = torch.randint(0, 1000, (1, 5)).to(device) # Mock prompt of length 5
    images = torch.randn(1, 3, 224, 224).to(device)          # Mock image
    
    # Mock token decoding function (Replace with tokenizer.decode)
    def mock_decode(token_ids):
        return [f"tok_{t.item()}" for t in token_ids[0]]

    # --- 3. GENERATION ---
    # Run the model's generate function (using your standard greedy/multinomial generate)
    output_tokens = model.generate(
        images=images, 
        start_token_id=start_tokens, 
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    
    # Combine prompt and generated tokens to match the sequence of routed tokens
    all_tokens = torch.cat([start_tokens, output_tokens], dim=1)
    decoded_tokens = mock_decode(all_tokens) # List of string words/tokens
    generated_text = " ".join(decoded_tokens)
    
    # --- 4. PROCESS HOOK DATA ---
    # Concatenate the lists of prefill + decode steps into a single sequence array per layer
    # Format per layer: shape (Total_Seq_Len, TOP_K)
    processed_history = {}
    for layer, history_list in routing_history.items():
        processed_history[layer] = np.concatenate(history_list, axis=0)

    # --- 5. VISUALIZATION 1: Expert Load Balance (Bar Chart) ---
    all_expert_selections = []
    for layer, assignments in processed_history.items():
        all_expert_selections.extend(assignments.flatten())
    
    expert_counts = np.bincount(all_expert_selections, minlength=NUM_EXPERTS)
    
    fig_balance = px.bar(
        x=[f"Expert {i}" for i in range(NUM_EXPERTS)], 
        y=expert_counts,
        labels={"x": "Expert ID", "y": "Number of Tokens Handled"},
        title="Global Expert Load Balancing",
        color=expert_counts,
        color_continuous_scale="Blues"
    )

    # --- 6. VISUALIZATION 2: Token-to-Expert Heatmap ---
    layer_names = list(processed_history.keys())
    
    # We will plot the Primary Expert (index 0 of top_k) as the color
    # And we will show both Primary and Secondary on Hover
    primary_experts = np.zeros((len(decoded_tokens), len(layer_names)))
    hover_text = np.empty((len(decoded_tokens), len(layer_names)), dtype=object)

    for col_idx, layer in enumerate(layer_names):
        assignments = processed_history[layer]
        for row_idx in range(len(decoded_tokens)):
            # Fallback in case seq length mismatches
            if row_idx < len(assignments):
                primary = assignments[row_idx, 0]
                secondary = assignments[row_idx, 1] if TOP_K > 1 else "N/A"
                primary_experts[row_idx, col_idx] = primary
                hover_text[row_idx, col_idx] = f"Token: '{decoded_tokens[row_idx]}'<br>Primary: Exp {primary}<br>Secondary: Exp {secondary}"
            else:
                primary_experts[row_idx, col_idx] = -1
                hover_text[row_idx, col_idx] = "No Data"

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=primary_experts,
        x=layer_names,
        y=decoded_tokens,
        text=hover_text,
        hoverinfo="text",
        colorscale="Turbo",
        zmin=0, zmax=NUM_EXPERTS-1,
        colorbar=dict(title="Expert ID")
    ))
    
    fig_heatmap.update_layout(
        title="Token-by-Token Routing Behavior",
        yaxis=dict(autorange="reversed"), # Read text top-to-bottom
        xaxis=dict(side="top"),
        height=max(400, len(decoded_tokens) * 20) # Auto-scale height based on text length
    )

    return generated_text, fig_balance, fig_heatmap

# --- 7. INITIALIZATION SCRIPT ---
def init_app():
    global model
    # Initialize your model here
    # model = SwinMoETex(vocab_size=32000).to(device)
    # model.load_state_dict(torch.load("path_to_weights.pt"))
    
    # Mocking the model for Gradio layout rendering (Remove this in production)
    import torch.nn as nn
    class MockRouter(nn.Module):
        def forward(self, x): return torch.randn(x.size(0), NUM_EXPERTS)
    class MockMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = MockRouter()
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.m1 = MockMoE()
            self.m2 = MockMoE()
        def generate(self, **kwargs): return torch.randint(0, 1000, (1, kwargs["max_new_tokens"]))
    model = MockModel()
    
    # Attach the listening hooks
    attach_hooks(model)

# --- 8. GRADIO UI LAYOUT ---
init_app()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 SwinMoETex: Mixture of Experts Visualizer")
    gr.Markdown("Upload an image and provide a prompt. See exactly which experts activate for which tokens during inference!")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="filepath", label="Input Image (Optional for Text-only)")
            txt_input = gr.Textbox(lines=2, label="Text Prompt", value="A close up of a")
            
            with gr.Row():
                max_tokens = gr.Slider(10, 200, value=50, step=1, label="Max New Tokens")
                temp_slider = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Temperature")
                
            gen_button = gr.Button("Generate & Analyze Routing", variant="primary")
            
        with gr.Column():
            out_text = gr.Textbox(lines=6, label="Generated Sequence")
            
    with gr.Row():
        out_bar = gr.Plot(label="Expert Load Balancing")
        
    with gr.Row():
        out_heatmap = gr.Plot(label="Routing Heatmap (Hover for info)")

    gen_button.click(
        fn=generate_and_visualize,
        inputs=[img_input, txt_input, max_tokens, temp_slider],
        outputs=[out_text, out_bar, out_heatmap]
    )

if __name__ == "__main__":
    demo.launch(share=True)