import transformer_engine.pytorch as te
import torch
    
    
class TinyTransformerModel(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, dropout: float):
        super().__init__()
        self.depth_dim = input_shape[1]
        self.features_dim = input_shape[2]
        self.temporal_dim = input_shape[0]
        self.inputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.outputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.base = 10000.0
        half_dim = (self.features_dim * self.depth_dim) // 2

        # Create position indices [0, 1, ..., T-1]
        position = torch.arange(self.temporal_dim, dtype=torch.float32).unsqueeze(1)  # (T, 1)

        # Compute the inverse frequencies
        div_term = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float32) * (-torch.log(torch.tensor(self.base)) / half_dim)
        )  # (half_dim,)

        # Compute the angles (T, half_dim)
        angles = position * div_term  # (T, half_dim)

        # Compute sin and cos
        sin = torch.sin(angles).requires_grad_(False)  # (T, half_dim)
        cos = torch.cos(angles).requires_grad_(False)  # (T, half_dim)
        sin = sin.unsqueeze(0).unsqueeze(-1)  # (1, T, 1, F//2)
        cos = cos.unsqueeze(0).unsqueeze(-1)

        # Register as buffers to ensure they are moved with the model and not trained
        self.register_buffer('sin', sin)  # (T, half_dim)
        self.register_buffer('cos', cos)  # (T, half_dim)

        #self.embedding_layer = te.Linear(self.features_dim*self.depth_dim, self.features_dim*self.depth_dim)
        #self.embedding_layer = torch.nn.Linear(self.features_dim*self.depth_dim, self.features_dim*self.depth_dim)
        #self.embedding_dropout = torch.nn.Dropout(dropout)
        #self.embedding_relu = torch.nn.ReLU()

        #self.output_fc = torch.nn.Linear(self.features_dim*self.depth_dim, self.features_dim*self.depth_dim)
        self.output_fc = te.Linear(self.features_dim*self.depth_dim, self.features_dim*self.depth_dim)
        self.output_dropout = torch.nn.Dropout(dropout)
        self.output_relu = torch.nn.ReLU()

        #self.positional_encoder = Summer(PositionalEncoding1D(self.features_dim*self.depth_dim))
        
        self.encoder1 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400, 
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder2 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder3 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder4 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder5 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder6 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder7 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400, 
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder8 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder9 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder10 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder11 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder12 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        """
        self.encoder13 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder14 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder15 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder16 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        
        self.encoder17 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder18 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder19 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder20 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")"""

    def apply_rotary_pos_emb(self, x, sin, cos) -> torch.Tensor:
        """
        Applies Rotary Positional Embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F).
            sin (torch.Tensor): Sine embeddings of shape (T, F/2).
            cos (torch.Tensor): Cosine embeddings of shape (T, F/2).

        Returns:
            torch.Tensor: Tensor with RoPE applied, shape (B, T, F).
        """
        # Ensure the feature dimension is even

        # Split the features into even and odd
        x_even = x[..., 0::2]  # (B, T, F/2)
        x_odd = x[..., 1::2]   # (B, T, F/2)

        # Apply rotation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Interleave the rotated even and odd features
        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1).reshape_as(x)  # (B, T, D, F)

        return x_rotated


    def forward(self, input):
        #x = x.view(-1, self.inputs_shape)
        #x = x.view(self.temporal_dim, self.depth_dim, self.features_dim)
        input = self.apply_rotary_pos_emb(input, self.sin, self.cos)  # (B, T, D, F)

        input = input.view(-1, self.temporal_dim, self.features_dim * self.depth_dim)
        #input = input + self.pe[:input.size(1)]
        
        #embedding = self.embedding_layer(input)
        #embedding = self.embedding_relu(embedding)
        #embedding = self.embedding_dropout(embedding)
        #x = self.positional_encoder(x)
        #print(f"Shape after positional encoding: {x.shape}")
        output = self.encoder1(input)
        output = self.encoder2(output)
        output = self.encoder3(output)
        output = self.encoder4(output)
        output = self.encoder5(output)
        output = self.encoder6(output)
        output = self.encoder7(output)
        output = self.encoder8(output)
        output = self.encoder9(output)
        output = self.encoder10(output)
        output = self.encoder11(output)
        output = self.encoder12(output)
        #output = self.encoder13(output)
        #output = self.encoder14(output)
        #output = self.encoder15(output)
        #output = self.encoder16(output)
        #output = self.encoder17(output)
        #output = self.encoder18(output)
        #output = self.encoder19(output)
        #output = self.encoder20(output)
        output = self.output_fc(output)
        output = self.output_relu(output)
        output = self.output_dropout(output)
        x = output.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)

        #x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        return x
    
class MediumTransformerModel(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, dropout: float):
        super().__init__()
        self.depth_dim = input_shape[1]
        self.features_dim = input_shape[2]
        self.temporal_dim = input_shape[0]
        self.inputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.outputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.base = 10000.0
        half_dim = (self.features_dim * self.depth_dim) // 2

        # Create position indices [0, 1, ..., T-1]
        position = torch.arange(self.temporal_dim, dtype=torch.float32).unsqueeze(1)  # (T, 1)

        # Compute the inverse frequencies
        div_term = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float32) * (-torch.log(torch.tensor(self.base)) / half_dim)
        )  # (half_dim,)

        # Compute the angles (T, half_dim)
        angles = position * div_term  # (T, half_dim)

        # Compute sin and cos
        sin = torch.sin(angles).requires_grad_(False)  # (T, half_dim)
        cos = torch.cos(angles).requires_grad_(False)  # (T, half_dim)
        sin = sin.unsqueeze(0).unsqueeze(-1)  # (1, T, 1, F//2)
        cos = cos.unsqueeze(0).unsqueeze(-1)

        # Register as buffers to ensure they are moved with the model and not trained
        self.register_buffer('sin', sin)  # (T, half_dim)
        self.register_buffer('cos', cos)  # (T, half_dim)

        self.embedding_layer = te.Linear(self.features_dim*self.depth_dim, self.features_dim*self.depth_dim)
        self.embedding_relu = torch.nn.ReLU()
        self.embedding_dropout = torch.nn.Dropout(dropout)
        
        self.output_fc = te.Linear(self.features_dim*self.depth_dim, self.features_dim*self.depth_dim)
        self.output_relu = torch.nn.ReLU()
        self.output_dropout = torch.nn.Dropout(dropout)

        self.encoder1 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400, 
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder2 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder3 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder4 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder5 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder6 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder7 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400, 
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder8 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder9 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder10 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder11 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder12 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder13 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder14 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder15 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder16 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        
        self.encoder17 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder18 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder19 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder20 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder21 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder22 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder23 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder24 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=6400,
                                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

    
    def apply_rotary_pos_emb(self, x, sin, cos) -> torch.Tensor:
        """
        Applies Rotary Positional Embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F).
            sin (torch.Tensor): Sine embeddings of shape (T, F/2).
            cos (torch.Tensor): Cosine embeddings of shape (T, F/2).

        Returns:
            torch.Tensor: Tensor with RoPE applied, shape (B, T, F).
        """
        # Ensure the feature dimension is even

        # Split the features into even and odd
        x_even = x[..., 0::2]  # (B, T, F/2)
        x_odd = x[..., 1::2]   # (B, T, F/2)

        # Apply rotation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Interleave the rotated even and odd features
        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1).reshape_as(x)  # (B, T, D, F)

        return x_rotated


    def forward(self, input):
        #x = x.view(-1, self.inputs_shape)
        #x = x.view(self.temporal_dim, self.depth_dim, self.features_dim)
        input = self.apply_rotary_pos_emb(input, self.sin, self.cos)  # (B, T, D, F)

        input = input.view(-1, self.temporal_dim, self.features_dim * self.depth_dim)
        #input = input + self.pe[:input.size(1)]
        
        embedding = self.embedding_layer(input)
        embedding = self.embedding_relu(embedding)
        embedding = self.embedding_dropout(embedding)
        #x = self.positional_encoder(x)
        #print(f"Shape after positional encoding: {x.shape}")
        output = self.encoder1(embedding)
        output = self.encoder2(output)
        output = self.encoder3(output)
        output = self.encoder4(output)
        output = self.encoder5(output)
        output = self.encoder6(output)
        output = self.encoder7(output)
        output = self.encoder8(output)
        output = self.encoder9(output)
        output = self.encoder10(output)
        output = self.encoder11(output)
        output = self.encoder12(output)
        output = self.encoder13(output)
        output = self.encoder14(output)
        output = self.encoder15(output)
        output = self.encoder16(output)
        output = self.encoder17(output)
        output = self.encoder18(output)
        output = self.encoder19(output)
        output = self.encoder20(output)
        output = self.encoder21(output)
        output = self.encoder22(output)
        output = self.encoder23(output)
        output = self.encoder24(output)

        output = self.output_fc(output)
        output = self.output_relu(output)
        output = self.output_dropout(output)
        x = output.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)

        #x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        return x
    
class DeepNarrowTransformerModel(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, dropout: float):
        super().__init__()
        self.depth_dim = input_shape[1]
        self.features_dim = input_shape[2]
        self.temporal_dim = input_shape[0]
        self.inputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.outputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.base = 10000.0
        half_dim = (self.features_dim * self.depth_dim) // 2

        # Create position indices [0, 1, ..., T-1]
        position = torch.arange(self.temporal_dim, dtype=torch.float32).unsqueeze(1)  # (T, 1)

        # Compute the inverse frequencies
        div_term = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float32) * (-torch.log(torch.tensor(self.base)) / half_dim)
        )  # (half_dim,)

        # Compute the angles (T, half_dim)
        angles = position * div_term  # (T, half_dim)

        # Compute sin and cos
        sin = torch.sin(angles).requires_grad_(False)  # (T, half_dim)
        cos = torch.cos(angles).requires_grad_(False)  # (T, half_dim)
        sin = sin.unsqueeze(0).unsqueeze(-1)  # (1, T, 1, F//2)
        cos = cos.unsqueeze(0).unsqueeze(-1)

        # Register as buffers to ensure they are moved with the model and not trained
        self.register_buffer('sin', sin)  # (T, half_dim)
        self.register_buffer('cos', cos)  # (T, half_dim)

        self.embedding_layer = te.Linear(self.features_dim*self.depth_dim, self.features_dim*self.depth_dim)
        self.embedding_relu = torch.nn.ReLU()
        self.embedding_dropout = torch.nn.Dropout(dropout)
        
        self.output_fc = te.Linear(self.features_dim*self.depth_dim, self.features_dim*self.depth_dim)
        self.output_relu = torch.nn.ReLU()
        self.output_dropout = torch.nn.Dropout(dropout)

        self.encoder1 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536, 
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder2 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder3 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder4 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder5 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder6 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder7 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536, 
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder8 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder9 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder10 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder11 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder12 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder13 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder14 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder15 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder16 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        
        self.encoder17 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder18 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder19 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder20 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder21 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder22 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder23 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder24 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        self.encoder25 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536, 
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder26 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder27 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder28 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder29 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder30 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder31 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536, 
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder32 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder33 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder34 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder35 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
        
        self.encoder36 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                            num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                            fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                            parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder37 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder38 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder39 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")

        self.encoder40 = te.TransformerLayer(hidden_size=self.features_dim*self.depth_dim, ffn_hidden_size=1536,
                                                    num_attention_heads=8, layer_type='encoder', hidden_dropout=dropout,
                                                    fuse_qkv_params=True, set_parallel_mode=True,  attn_input_format="bshd", 
                                                    parallel_attention_mlp=True, attention_dropout=dropout, self_attn_mask_type="no_mask")
    
    def apply_rotary_pos_emb(self, x, sin, cos) -> torch.Tensor:
        """
        Applies Rotary Positional Embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F).
            sin (torch.Tensor): Sine embeddings of shape (T, F/2).
            cos (torch.Tensor): Cosine embeddings of shape (T, F/2).

        Returns:
            torch.Tensor: Tensor with RoPE applied, shape (B, T, F).
        """
        # Ensure the feature dimension is even

        # Split the features into even and odd
        x_even = x[..., 0::2]  # (B, T, F/2)
        x_odd = x[..., 1::2]   # (B, T, F/2)

        # Apply rotation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Interleave the rotated even and odd features
        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1).reshape_as(x)  # (B, T, D, F)

        return x_rotated
    
    def forward(self, input):
        #x = x.view(-1, self.inputs_shape)
        #x = x.view(self.temporal_dim, self.depth_dim, self.features_dim)
        input = self.apply_rotary_pos_emb(input, self.sin, self.cos)  # (B, T, D, F)

        input = input.view(-1, self.temporal_dim, self.features_dim * self.depth_dim)
        #input = input + self.pe[:input.size(1)]
        
        embedding = self.embedding_layer(input)
        embedding = self.embedding_relu(embedding)
        embedding = self.embedding_dropout(embedding)
        #x = self.positional_encoder(x)
        #print(f"Shape after positional encoding: {x.shape}")
        output = self.encoder1(embedding)
        output = self.encoder2(output)
        output = self.encoder3(output)
        output = self.encoder4(output)
        output = self.encoder5(output)
        output = self.encoder6(output)
        output = self.encoder7(output)
        output = self.encoder8(output)
        output = self.encoder9(output)
        output = self.encoder10(output)
        output = self.encoder11(output)
        output = self.encoder12(output)
        output = self.encoder13(output)
        output = self.encoder14(output)
        output = self.encoder15(output)
        output = self.encoder16(output)
        output = self.encoder17(output)
        output = self.encoder18(output)
        output = self.encoder19(output)
        output = self.encoder20(output)
        output = self.encoder21(output)
        output = self.encoder22(output)
        output = self.encoder23(output)
        output = self.encoder24(output)
        output = self.encoder25(output)
        output = self.encoder26(output)
        output = self.encoder27(output)
        output = self.encoder28(output)
        output = self.encoder29(output)
        output = self.encoder30(output)
        output = self.encoder31(output)
        output = self.encoder32(output)
        output = self.encoder33(output)
        output = self.encoder34(output)
        output = self.encoder35(output)
        output = self.encoder36(output)
        output = self.encoder37(output)
        output = self.encoder38(output)
        output = self.encoder39(output)
        output = self.encoder40(output)

        output = self.output_fc(output)
        output = self.output_relu(output)
        output = self.output_dropout(output)
        x = output.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)

        #x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        return x
    
class DeepNarrowTransformerModelPT(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, dropout: float):
        super().__init__()
        self.depth_dim = input_shape[1]
        self.features_dim = input_shape[2]
        self.temporal_dim = input_shape[0]
        self.inputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.outputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.base = 10000.0
        half_dim = (self.features_dim * self.depth_dim) // 2
        self.out_channels = 128


        # Create position indices [0, 1, ..., T-1]
        position = torch.arange(self.temporal_dim, dtype=torch.float16).unsqueeze(1)  # (T, 1)

        # Compute the inverse frequencies
        div_term = torch.exp(
            torch.arange(0, half_dim, dtype=torch.float16) * (-torch.log(torch.tensor(self.base)) / half_dim)
        )  # (half_dim,)

        # Compute the angles (T, half_dim)
        angles = position * div_term  # (T, half_dim)

        # Compute sin and cos
        sin = torch.sin(angles).requires_grad_(False)  # (T, half_dim)
        cos = torch.cos(angles).requires_grad_(False)  # (T, half_dim)
        sin = sin.unsqueeze(0).unsqueeze(-1)  # (1, T, 1, F//2)
        cos = cos.unsqueeze(0).unsqueeze(-1)

        # Register as buffers to ensure they are moved with the model and not trained
        self.register_buffer('sin', sin)  # (T, half_dim)
        self.register_buffer('cos', cos)  # (T, half_dim)

        self.conv1 = torch.nn.Conv2d(in_channels=self.temporal_dim, out_channels=self.out_channels, kernel_size=2, stride=2, dilation=1)
        #self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv1_relu = torch.nn.ReLU()
        self.conv1_dropout = torch.nn.Dropout(dropout)

        #self.embedding_layer = torch.nn.Linear(self.features_dim*self.depth_dim, self.features_dim*self.depth_dim)
        #self.embedding_relu = torch.nn.ReLU()
        #self.embedding_dropout = torch.nn.Dropout(dropout)
        
        self.output_fc = torch.nn.Linear(int(int(self.depth_dim/2)*self.out_channels), 32)
        self.output_relu = torch.nn.ReLU()
        self.output_dropout = torch.nn.Dropout(dropout)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=(int(self.depth_dim/2)),
            dim_feedforward=512,
            nhead=8,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=12)
    
    def apply_rotary_pos_emb(self, x, sin, cos) -> torch.Tensor:
        """
        Applies Rotary Positional Embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, F).
            sin (torch.Tensor): Sine embeddings of shape (T, F/2).
            cos (torch.Tensor): Cosine embeddings of shape (T, F/2).

        Returns:
            torch.Tensor: Tensor with RoPE applied, shape (B, T, F).
        """
        # Ensure the feature dimension is even

        # Split the features into even and odd
        x_even = x[..., 0::2]  # (B, T, F/2)
        x_odd = x[..., 1::2]   # (B, T, F/2)

        # Apply rotation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Interleave the rotated even and odd features
        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1).reshape_as(x)  # (B, T, D, F)

        return x_rotated
    
    def forward(self, input):
        #x = x.view(-1, self.inputs_shape)
        #x = x.view(self.temporal_dim, self.depth_dim, self.features_dim)
        input = self.apply_rotary_pos_emb(input, self.sin, self.cos)  # (B, T, D, F)
        input = self.conv1(input)
        input = self.conv1_relu(input)
        input = self.conv1_dropout(input)
        input = input.view(-1, self.out_channels, int(self.depth_dim/2))

        #print(f"Shape after conv1: {input.shape}")

        #input = torch.permute(input, (0, 2, 1, 3))

        #input = input.view(-1, self.temporal_dim, self.features_dim * self.depth_dim)
        #input = input + self.pe[:input.size(1)]
        
        #embedding = self.embedding_layer(input)
        #embedding = self.embedding_relu(embedding)
        #embedding = self.embedding_dropout(embedding)
        #x = self.positional_encoder(x)
        #print(f"Shape after positional encoding: {x.shape}")
        output = self.encoder(input)
        output = output.view(-1, int(self.depth_dim/2)*self.out_channels)

        output = self.output_fc(output)
        output = self.output_relu(output)
        output = self.output_dropout(output)
        #x = output.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)

        #x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        return output

class StateEncoder(torch.nn.Module):
    def __init__(self, state_features_dim: int, temporal_dim: int, output_shape: tuple, dropout: float):
        super().__init__()
        self.state_features_dim = state_features_dim
        self.temporal_dim = temporal_dim
        self.output_shape = output_shape
        self.dropout = dropout

        self.embedding_layer1 = torch.nn.Linear(self.state_features_dim, self.state_features_dim)
        self.embedding_relu1 = torch.nn.ReLU()
        self.embedding_dropout1 = torch.nn.Dropout(dropout)
        
        self.embedding_layer2 = torch.nn.Linear(self.state_features_dim, self.state_features_dim)
        self.embedding_relu2 = torch.nn.ReLU()
        self.embedding_dropout2 = torch.nn.Dropout(dropout)
        
        self.embedding_layer3 = torch.nn.Linear(self.state_features_dim, self.state_features_dim)
        self.embedding_relu3 = torch.nn.ReLU()
        self.embedding_dropout3 = torch.nn.Dropout(dropout)
        
        self.output_fc = torch.nn.Linear(self.state_features_dim, self.state_features_dim*2)
        self.output_relu = torch.nn.ReLU()
        self.output_dropout = torch.nn.Dropout(dropout)
    
    def forward(self, input):
        embedding = self.embedding_layer1(input)
        embedding = self.embedding_relu1(embedding)
        embedding = self.embedding_dropout1(embedding)

        embedding = self.embedding_layer2(embedding)
        embedding = self.embedding_relu2(embedding)
        embedding = self.embedding_dropout2(embedding)

        embedding = self.embedding_layer3(embedding)
        embedding = self.embedding_relu3(embedding)
        embedding = self.embedding_dropout3(embedding)

        output = self.output_fc(embedding)
        output = self.output_relu(output)
        output = self.output_dropout(output)

        return output

class PPOModel(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, dropout: float, state_features_dim: int, ob_encoder: DeepNarrowTransformerModelPT):
        super().__init__()
        self.temporal_dim = input_shape[0]
        self.ob_encoder: DeepNarrowTransformerModelPT = ob_encoder
        self.state_encoder = StateEncoder(state_features_dim=state_features_dim, temporal_dim=input_shape[0], output_shape=output_shape, dropout=dropout)
        
        self.policy_fc1 = torch.nn.Linear(64, 128)
        self.policy_fc1_activation = torch.nn.ReLU()
        self.policy_fc1_dropout = torch.nn.Dropout(dropout)

        self.policy_fc2 = torch.nn.Linear(128, 256)
        self.policy_fc2_activation = torch.nn.ReLU()
        self.policy_fc2_dropout = torch.nn.Dropout(dropout)

        self.policy_fc3 = torch.nn.Linear(256, 512)
        self.policy_fc3_activation = torch.nn.ReLU()
        self.policy_fc3_dropout = torch.nn.Dropout(dropout)

        self.policy_fc4 = torch.nn.Linear(512, 256)
        self.policy_fc4_activation = torch.nn.ReLU()
        self.policy_fc4_dropout = torch.nn.Dropout(dropout)


        self.policy_fc5 = torch.nn.Linear(256, 128)
        self.policy_fc5_activation = torch.nn.ReLU()
        self.policy_fc5_dropout = torch.nn.Dropout(dropout)

        self.policy_fc6 = torch.nn.Linear(128, 64)
        self.policy_fc6_activation = torch.nn.ReLU()
        self.policy_fc6_dropout = torch.nn.Dropout(dropout)

        self.policy_output = torch.nn.Linear(64, 3)
        self.value_output = torch.nn.Linear(64, 1)

    def forward(self, ob_input, state_input):
        ob_output = self.ob_encoder(ob_input)
        #print(f"ob encoder shape: {ob_output.shape}")
        #ob_output = ob_output.view(-1, (self.ob_encoder.depth_dim * self.ob_encoder.features_dim * self.ob_encoder.temporal_dim))
        #ob_output = self.ob_state_reducer(ob_output)
        state_output = self.state_encoder(state_input)
        #print(f"state encoder shape: {state_output.shape}")
        x = torch.cat((ob_output, state_output), dim=-1)

        x = self.policy_fc1(x)
        x = self.policy_fc1_activation(x)
        x = self.policy_fc1_dropout(x)

        x = self.policy_fc2(x)
        x = self.policy_fc2_activation(x)
        x = self.policy_fc2_dropout(x)

        x = self.policy_fc3(x)
        x = self.policy_fc3_activation(x)
        x = self.policy_fc3_dropout(x)

        x = self.policy_fc4(x)
        x = self.policy_fc4_activation(x)
        x = self.policy_fc4_dropout(x)

        x = self.policy_fc5(x)
        x = self.policy_fc5_activation(x)
        x = self.policy_fc5_dropout(x)

        x = self.policy_fc6(x)
        x = self.policy_fc6_activation(x)
        x = self.policy_fc6_dropout(x)

        x = x.flatten(start_dim=1)
        value = self.value_output(x)
        x = self.policy_output(x)
        #x = torch.nn.functional.softmax(x)

        return x, value