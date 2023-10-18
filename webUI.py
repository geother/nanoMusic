from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile
import os
from tqdm import tqdm
import pickle
from contextlib import nullcontext
import torch
import numpy as np
from model import GPTConfig, GPT
import streamlit as st
from streamlit_extras.grid import grid
import pretty_midi
import seaborn as sns
import matplotlib.pyplot as plt

device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
seed = 1337


@st.cache_resource
def get_model():
    model_list = os.listdir("out")
    if "pop42k.pt" not in model_list:
        print(
            "load model error: please download the model from \nhttps://box.nju.edu.cn/f/5d48cc09350e4116adfe/?dl=1"
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    REMIconfig = TokenizerConfig(nb_velocities=16, use_chords=False, use_programs=True)
    tokenizer = REMI(REMIconfig)

    out_dir = "out"  # ignored if init_from is not 'resume'
    use_model = "pop42k.pt"
    ckpt_path = os.path.join(out_dir, use_model)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return tokenizer, model


tokenizer, model = get_model()


def run(max_new_tokens, temperature, top_k, col):
    num_samples = 1  # number of samples to draw
    midi = MidiFile("test/IN/qby_test.mid")
    tokens = tokenizer(midi).ids
    tokens = torch.tensor(tokens, requires_grad=False, device=device)[None, ...]

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = (
                    model.generate(tokens, max_new_tokens, temperature=temperature, top_k=top_k)
                    .squeeze()
                    .cpu()
                    .numpy()
                )

    converted_back_midi = tokenizer(y)
    converted_back_midi.dump(f"test/OUT/stest.mid")
    midi_array = pretty_midi.PrettyMIDI(f"test/OUT/stest.mid")
    midi_array = midi_array.get_piano_roll(fs=16)
    print(midi_array.shape)
    fig, ax = plt.subplots()
    sns.heatmap(midi_array[32:110, :], cbar=False, xticklabels="", yticklabels="")
    with col:
        st.write(fig)


col1, col2 = st.columns((1, 1))

with col1:
    st.subheader("nanoMusic")
    midi_file = st.text_input("midi file name(under test/IN)")
    max_new_tokens = st.slider("max_new_tokens", 100, 2048, 512)
    temperature = st.slider("temperature", 0.0, 2.0, 0.89)
    top_k = st.slider("top_k", 10, 200, 89)
if midi_file:
    out_midi, midi_img = run(max_new_tokens, temperature, top_k, col2)
