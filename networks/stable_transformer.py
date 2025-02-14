import torch
import torch.nn.functional as F


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super(PositionalEmbedding, self).__init__()

        self.dim = dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class PositionwiseFF(torch.nn.Module):
    def __init__(self, d_input, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.dropout = dropout
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_input, d_inner),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_inner, d_input),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input_):
        ff_out = self.ff(input_)
        return ff_out


class GatingMechanism(torch.nn.Module):
    def __init__(self, d_input, bg=0.1):
        super(GatingMechanism, self).__init__()
        self.Wr = torch.nn.Linear(d_input, d_input)
        self.Ur = torch.nn.Linear(d_input, d_input)
        self.Wz = torch.nn.Linear(d_input, d_input)
        self.Uz = torch.nn.Linear(d_input, d_input)
        self.Wg = torch.nn.Linear(d_input, d_input)
        self.Ug = torch.nn.Linear(d_input, d_input)
        self.bg = bg

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g


class MultiHeadAttentionXL(torch.nn.Module):
    def __init__(self, d_input, d_inner, n_heads=4, dropout=0.1, dropouta=0.0):
        super(MultiHeadAttentionXL, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.n_heads = n_heads

        # Linear transformation for keys & values for all heads at once for efficiency.
        # 2 for keys & values.
        self.linear_kv = torch.nn.Linear(d_input, (d_inner * n_heads * 2), bias=False)
        # for queries (will not be concatenated with memorized states so separate).
        self.linear_q = torch.nn.Linear(d_input, d_inner * n_heads, bias=False)

        # for positional embeddings.
        self.linear_p = torch.nn.Linear(d_input, d_inner * n_heads, bias=False)
        self.scale = 1 / (d_inner ** 0.5)  # for scaled dot product attention
        self.dropa = torch.nn.Dropout(dropouta)

        self.lout = torch.nn.Linear(d_inner * n_heads, d_input, bias=False)
        self.dropo = torch.nn.Dropout(dropout)

    def _rel_shift(self, x):
        # x shape: [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        zero_pad = torch.zeros(
            (x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype
        )
        return (
            torch.cat([zero_pad, x], dim=1)
            .view(x.size(1) + 1, x.size(0), *x.size()[2:])[1:]
            .view_as(x)
        )

    def forward(self, input_, pos_embs, memory, u, v, mask=None):
        """
        + pos_embs: positional embeddings passed separately to handle relative positions.
        + Arguments
            - input: torch.FloatTensor, shape - (seq, bs, self.d_input) = (20, 5, 8)
            - pos_embs: torch.FloatTensor, shape - (seq + prev_seq, bs, self.d_input) = (40, 1, 8)
            - memory: torch.FloatTensor, shape - (prev_seq, b, d_in) = (20, 5, 8)
            - u: torch.FloatTensor, shape - (num_heads, inner_dim) = (3 x )
            - v: torch.FloatTensor, shape - (num_heads, inner_dim)
            - mask: torch.FloatTensor, Optional = (20, 40, 1)

        + Returns
            - output: torch.FloatTensor, shape - (seq, bs, self.d_input)

        + symbols representing shape of the tensors
            - cs: current sequence length, b: batch, H: no. of heads
            - d: inner dimension, ps: previous sequence length
        """
        cur_seq = input_.shape[0]
        prev_seq = memory.shape[0]
        H, d = self.n_heads, self.d_inner
        # concat memory across sequence dimension
        # input_with_memory = [seq + prev_seq x B x d_input] = [40 x 5 x 8]
        memory = memory.to(input_.device)
        if memory.size(1) != input_.size(1):
            print("memory shape: ", memory.shape)
            print("input_ shape: ", input_.shape)
            # throw error
            raise ValueError("Memory and input batch size mismatch ")
        if memory.size(2) != input_.size(2):
            print("memory shape: ", memory.shape)
            print("input_ shape: ", input_.shape)
            # throw error
            raise ValueError("Memory and input feature size mismatch ")

        input_with_memory = torch.cat([memory, input_], dim=0)

        # k_tfmd, v_tfmd = [seq + prev_seq x B x n_heads.d_head_inner], [seq + prev_seq x B x n_heads.d_head_inner]
        k_tfmd, v_tfmd = torch.chunk(
            self.linear_kv(input_with_memory),
            2,
            dim=-1,
        )
        # q_tfmd = [seq x B x n_heads.d_head_inner] = [20 x 5 x 96]
        q_tfmd = self.linear_q(input_)

        _, bs, _ = q_tfmd.shape
        assert bs == k_tfmd.shape[1]

        # content_attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        content_attn = torch.einsum(
            "ibhd,jbhd->ijbh",
            (
                (q_tfmd.view(cur_seq, bs, H, d) + u),
                k_tfmd.view(cur_seq + prev_seq, bs, H, d),
            ),
        )

        # p_tfmd: [seq + prev_seq x 1 x n_heads.d_head_inner] = [40 x 1 x 96]
        p_tfmd = self.linear_p(pos_embs)
        # position_attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        position_attn = torch.einsum(
            "ibhd,jhd->ijbh",
            (
                (q_tfmd.view(cur_seq, bs, H, d) + v),
                p_tfmd.view(cur_seq + prev_seq, H, d),
            ),
        )

        position_attn = self._rel_shift(position_attn)
        # attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        attn = content_attn + position_attn

        if mask is not None and mask.any().item():
            # fills float('-inf') where mask is True.
            attn = attn.masked_fill(mask[..., None], -float("inf"))
        # rescale to prevent values from exploding.
        # normalize across the value sequence dimension.
        attn = torch.softmax(attn * self.scale, dim=1)
        # attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        attn = self.dropa(attn)

        # attn_weighted_values = [curr x B x n_heads.d_inner] = [20 x 5 x 96]
        attn_weighted_values = (
            torch.einsum(
                "ijbh,jbhd->ibhd",
                (
                    attn,  # (cs, cs + ps, b, H)
                    v_tfmd.view(cur_seq + prev_seq, bs, H, d),  # (cs + ps, b, H, d)
                ),
            )  # (cs, b, H, d)
            .contiguous()  # we need to change the memory layout to make `view` work
            .view(cur_seq, bs, H * d)
        )  # (cs, b, H * d)

        # output = [curr x B x d_input] = [20 x 5 x 8]
        output = self.dropo(self.lout(attn_weighted_values))
        return output


class StableTransformerEncoderLayerXL(torch.nn.Module):
    def __init__(
        self,
        n_heads,
        d_input,
        d_head_inner,
        d_ff_inner,
        dropout,
        gating=True,
        dropouta=0.0,
    ):
        super(StableTransformerEncoderLayerXL, self).__init__()

        self.gating = gating
        self.gate1 = GatingMechanism(d_input)
        self.gate2 = GatingMechanism(d_input)
        self.mha = MultiHeadAttentionXL(
            d_input,
            d_head_inner,
            n_heads=n_heads,
            dropout=dropout,
            dropouta=dropouta,
        )
        self.ff = PositionwiseFF(d_input, d_ff_inner, dropout)
        self.norm1 = torch.nn.LayerNorm(d_input)
        self.norm2 = torch.nn.LayerNorm(d_input)

    def forward(self, input_, pos_embs, u, v, mask=None, mems=None):
        src2 = self.norm1(input_)
        src2 = self.mha(src2, pos_embs, mems, u, v, mask=mask)
        src = self.gate1(input_, src2) if self.gating else input_ + src2
        src2 = self.ff(self.norm2(src))
        src = self.gate2(src, src2) if self.gating else src + src2
        return src


class StableTransformerXL(torch.nn.Module):
    def __init__(
        self,
        d_input,
        n_layers,
        n_heads,
        d_head_inner,
        d_ff_inner,
        dropout=0.1,
        dropouta=0.0,
        mem_len=100,
    ):
        super(StableTransformerXL, self).__init__()

        (
            self.n_layers,
            self.n_heads,
            self.d_input,
            self.d_head_inner,
            self.d_ff_inner,
        ) = (n_layers, n_heads, d_input, d_head_inner, d_ff_inner)

        self.pos_embs = PositionalEmbedding(d_input)
        self.drop = torch.nn.Dropout(dropout)
        self.mem_len = mem_len
        self.layers = torch.nn.ModuleList(
            [
                StableTransformerEncoderLayerXL(
                    n_heads,
                    d_input,
                    d_head_inner=d_head_inner,
                    d_ff_inner=d_ff_inner,
                    dropout=dropout,
                    dropouta=dropouta,
                )
                for _ in range(n_layers)
            ]
        )

        # u and v are global parameters: maybe changing these to per-head parameters might help performance?
        self.u, self.v = (
            # [n_heads x d_head_inner] = [3 x 32]
            torch.nn.Parameter(torch.zeros(self.n_heads, self.d_head_inner)),
            torch.nn.Parameter(torch.zeros(self.n_heads, self.d_head_inner)),
        )

    def init_memory(self, batch_size, device=torch.device("cpu")):
        # Create empty memories with shape [0, batch_size, d_input]
        return [
            torch.empty(0, batch_size, self.d_input, device=device)
            for _ in range(self.n_layers + 1)
        ]

    def update_memory(self, previous_memory, hidden_states):
        """
        + Arguments
            - previous_memory: List[torch.FloatTensor],
            - hidden_states: List[torch.FloatTensor]
        """
        assert len(hidden_states) == len(previous_memory)
        mem_len, seq_len = previous_memory[0].size(0), hidden_states[0].size(0)
        # mem_len, seq_len = 3, hidden_states[0].size(0)
        # print(mem_len, seq_len)

        with torch.no_grad():
            new_memory = []
            end_idx = mem_len + seq_len
            beg_idx = max(0, end_idx - self.mem_len)
            for m, h in zip(previous_memory, hidden_states):
                m = m.to(h.device)
                cat = torch.cat([m, h], dim=0)
                new_memory.append(cat[beg_idx:end_idx].detach())
        return new_memory

    def forward(self, inputs, memory=None, src_key_padding_mask=None):
        """
        Arguments:
            inputs: Tensor of shape (T, B, d_input) -- current sequence tokens.
            memory: Optional list of Tensors, each of shape (mem_len, B, d_input).
            src_key_padding_mask: Optional Bool Tensor of shape (B, T) where True indicates a pad token.
        """
        if memory is None:
            memory = self.init_memory(inputs.shape[1], inputs.device)
        assert len(memory) == len(self.layers) + 1

        cur_seq, bs = inputs.shape[:2]
        prev_seq = memory[0].size(0)

        # Create the causal (future) mask.
        # This mask has shape (cur_seq, cur_seq+prev_seq, 1) and masks out tokens that should not be attended to.
        causal_mask = torch.triu(
            torch.ones((cur_seq, cur_seq + prev_seq), device=inputs.device, dtype=torch.bool),
            diagonal=1 + prev_seq,
        )[..., None]  # shape: (cur_seq, cur_seq+prev_seq, 1)

        # If a src_key_padding_mask is provided, create a mask for padded positions.
        if src_key_padding_mask is not None:
            # src_key_padding_mask is assumed to be of shape (B, cur_seq)
            # Memory tokens are assumed to be always valid (i.e. not padded).
            pad_mask_memory = torch.zeros((bs, prev_seq), dtype=torch.bool, device=inputs.device)
            # Concatenate: first prev_seq positions (memory) are False, then current sequence.
            key_padding_mask = torch.cat([pad_mask_memory, src_key_padding_mask], dim=1)  # shape: (B, prev_seq+cur_seq)
            # Reshape to combine with the causal mask:
            # We need a mask of shape (cur_seq, cur_seq+prev_seq, B)
            key_padding_mask = key_padding_mask.transpose(0, 1).unsqueeze(0).expand(cur_seq, -1, bs)
            # Also expand the causal mask to (cur_seq, cur_seq+prev_seq, B)
            causal_mask = causal_mask.expand(cur_seq, cur_seq + prev_seq, bs)
            # Combine the two masks: a position is masked if it is either a future token or a padded token.
            final_mask = causal_mask | key_padding_mask
        else:
            # Expand causal_mask to include the batch dimension.
            final_mask = causal_mask.expand(cur_seq, cur_seq + prev_seq, bs)

        pos_ips = torch.arange(cur_seq + prev_seq - 1, -1, -1.0, dtype=torch.float).to(inputs.device)
        # Obtain positional embeddings; shape: (cur_seq+prev_seq, 1, d_input)
        pos_embs = self.drop(self.pos_embs(pos_ips))
        if self.d_input % 2 != 0:
            pos_embs = pos_embs[:, :, :-1]

        hidden_states = [inputs]
        layer_out = inputs
        # Pass the combined mask to each encoder layer.
        for mem, layer in zip(memory, self.layers):
            layer_out = layer(
                layer_out,
                pos_embs,
                self.u,
                self.v,
                mask=final_mask,
                mems=mem,
            )
            hidden_states.append(layer_out)

        memory = self.update_memory(memory, hidden_states)
        return {"logits": layer_out, "memory": memory}