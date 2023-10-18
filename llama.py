import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import pytorch_lightning as pl
from transformers import LlamaModel, LlamaConfig

import numpy as np


class MIDITokenizer:
    def __init__(self):
        self.vocab_size = 0

        def allocate_ids(size):
            ids = [self.vocab_size + i for i in range(size)]
            self.vocab_size += size
            return ids

        self.pad_id = allocate_ids(1)[0]
        self.bos_id = allocate_ids(1)[0]
        self.eos_id = allocate_ids(1)[0]
        self.events = {
            "note": ["time1", "time2", "track", "duration", "channel", "pitch", "velocity"],
            "patch_change": ["time1", "time2", "track", "channel", "patch"],
            "control_change": ["time1", "time2", "track", "channel", "controller", "value"],
            "set_tempo": ["time1", "time2", "track", "bpm"],
        }
        self.event_parameters = {
            "time1": 128, "time2": 16, "duration": 2048, "track": 128, "channel": 16, "pitch": 128, "velocity": 128,
            "patch": 128, "controller": 128, "value": 128, "bpm": 256
        }
        self.event_ids = {e: allocate_ids(1)[0] for e in self.events.keys()}
        self.id_events = {i: e for e, i in self.event_ids.items()}
        self.parameter_ids = {p: allocate_ids(s) for p, s in self.event_parameters.items()}
        self.max_token_seq = max([len(ps) for ps in self.events.values()]) + 1


class MIDIModel(pl.LightningModule):
    def __init__(self, tokenizer: MIDITokenizer, n_layer=12, n_head=16, n_embd=1024, n_inner=4096, flash=False,
                 *args, **kwargs):
        super(MIDIModel, self).__init__()
        self.tokenizer = tokenizer
        self.net = LlamaModel(LlamaConfig(vocab_size=tokenizer.vocab_size,
                                          hidden_size=n_embd, num_attention_heads=n_head,
                                          num_hidden_layers=n_layer, intermediate_size=n_inner,
                                          pad_token_id=tokenizer.pad_id, max_position_embeddings=4096))
        self.net_token = LlamaModel(LlamaConfig(vocab_size=tokenizer.vocab_size,
                                                hidden_size=n_embd, num_attention_heads=n_head // 4,
                                                num_hidden_layers=n_layer // 4, intermediate_size=n_inner // 4,
                                                pad_token_id=tokenizer.pad_id, max_position_embeddings=4096))
        if flash:
            self.net = self.net.to_bettertransformer()
            self.net_token = self.net_token.to_bettertransformer()
        self.lm_head = nn.Linear(n_embd, tokenizer.vocab_size, bias=False)

    def forward_token(self, hidden_state, x=None):
        """

        :param hidden_state: (batch_size, n_embd)
        :param x: (batch_size, token_sequence_length)
        :return: (batch_size, 1 + token_sequence_length, vocab_size)
        """
        hidden_state = hidden_state.unsqueeze(1)  # (batch_size, 1, n_embd)
        if x is not None:
            x = self.net_token.embed_tokens(x)
            hidden_state = torch.cat([hidden_state, x], dim=1)
        hidden_state = self.net_token.forward(inputs_embeds=hidden_state).last_hidden_state
        return self.lm_head(hidden_state)

    def forward(self, x):
        """
        :param x: (batch_size, midi_sequence_length, token_sequence_length)
        :return: hidden (batch_size, midi_sequence_length, n_embd)
        """

        # merge token sequence
        x = self.net.embed_tokens(x)
        x = x.sum(dim=-2)
        x = self.net.forward(inputs_embeds=x)
        return x.last_hidden_state

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params



tk = MIDITokenizer()

model = MIDIModel(tk)

print("number of parameters: %.2fM" % (model.get_num_params()/1e6,))