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

        position = torch.arange(self.temporal_dim).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.features_dim*self.depth_dim, 2) * (-torch.log((torch.tensor(10000.0))) / self.features_dim*self.depth_dim))
        pe = torch.zeros(1, self.temporal_dim, self.features_dim*self.depth_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.embedding_layer = te.Linear(self.features_dim*self.depth_dim, self.features_dim*self.depth_dim)
        #self.embedding_layer = torch.nn.Linear(self.features_dim*self.depth_dim, self.features_dim*self.depth_dim)
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.embedding_relu = torch.nn.ReLU()

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


    def forward(self, input: torch.Tensor):
        #x = x.view(-1, self.inputs_shape)
        #x = x.view(self.temporal_dim, self.depth_dim, self.features_dim)
        self.input = input.view(-1, self.temporal_dim, self.features_dim * self.depth_dim)
        self.input = self.input + self.pe[:input.size(1)]
        
        self.embedding = self.embedding_layer(self.input)
        self.embedding = self.embedding_relu(self.embedding)
        self.embedding = self.embedding_dropout(self.embedding)
        #x = self.positional_encoder(x)
        #print(f"Shape after positional encoding: {x.shape}")
        self.output = self.encoder1(self.embedding)
        self.output = self.encoder2(self.output)
        self.output = self.encoder3(self.output)
        self.output = self.encoder4(self.output)
        self.output = self.encoder5(self.output)
        self.output = self.encoder6(self.output)
        self.output = self.encoder7(self.output)
        self.output = self.encoder8(self.output)
        self.output = self.encoder9(self.output)
        self.output = self.encoder10(self.output)
        self.output = self.encoder11(self.output)
        self.output = self.encoder12(self.output)
        self.output = self.encoder13(self.output)
        self.output = self.encoder14(self.output)
        self.output = self.encoder15(self.output)
        self.output = self.encoder16(self.output)
        
        self.output = self.output_fc(self.output)
        self.output = self.output_relu(self.output)
        self.output = self.output_dropout(self.output)
        x = self.output.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)


        '''
        self.output1 = self.transformer1(self.embedding)
        self.output2 = self.transformer2(self.output1+self.embedding)
        self.output3 = self.transformer3(self.output1+self.output2+self.embedding)
        self.output4 = self.transformer4(self.output1+self.output2+self.output3+self.embedding)
        self.output5 = self.transformer5(self.output1+self.output2+self.output3+self.output4+self.embedding)
        self.output6 = self.transformer6(self.output1+self.output2+self.output3+self.output4+self.output5+self.embedding)

        self.output7 = self.output_fc(self.output1+self.output2+self.output3+self.output4+self.output5+self.output6+self.embedding)
        self.output7 = self.output_dropout(self.output7)
        self.output7 = self.output_relu(self.output7)
        x = self.output7.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        
        '''
        #x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        return x