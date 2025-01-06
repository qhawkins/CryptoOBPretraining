import torch
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

class SmallFCModel(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, dropout: float):
        super().__init__()
        self.depth_dim = input_shape[0]
        self.features_dim = input_shape[1]
        self.temporal_dim = input_shape[2]
        self.inputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.outputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.fc1 = torch.nn.Linear(self.inputs_shape, 4096)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout)
        self.fc3 = torch.nn.Linear(4096, 4096)
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(dropout)
        self.fc4 = torch.nn.Linear(4096, 4096)
        self.relu4 = torch.nn.ReLU()
        self.dropout4 = torch.nn.Dropout(dropout)
        self.fc5 = torch.nn.Linear(4096, 4096)
        self.relu5 = torch.nn.ReLU()
        self.dropout5 = torch.nn.Dropout(dropout)
        self.fc6 = torch.nn.Linear(4096, 4096)
        self.relu6 = torch.nn.ReLU()
        self.dropout6 = torch.nn.Dropout(dropout)
        self.fc7 = torch.nn.Linear(4096, self.outputs_shape)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.inputs_shape)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.dropout5(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        return x

class MediumFCModel(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, dropout: float):
        super().__init__()
        self.depth_dim = input_shape[0]
        self.features_dim = input_shape[1]
        self.temporal_dim = input_shape[2]
        self.inputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.outputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.fc1 = torch.nn.Linear(self.inputs_shape, 4096)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout)
        self.fc3 = torch.nn.Linear(4096, 8192)
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(dropout)
        self.fc4 = torch.nn.Linear(8192, 8192)
        self.relu4 = torch.nn.ReLU()
        self.dropout4 = torch.nn.Dropout(dropout)
        self.fc5 = torch.nn.Linear(8192, 8192)
        self.relu5 = torch.nn.ReLU()
        self.dropout5 = torch.nn.Dropout(dropout)
        self.fc6 = torch.nn.Linear(8192, 8192)
        self.relu6 = torch.nn.ReLU()
        self.dropout6 = torch.nn.Dropout(dropout)
        self.fc7 = torch.nn.Linear(8192, 4096)
        self.relu7 = torch.nn.ReLU()
        self.dropout7 = torch.nn.Dropout(dropout)
        self.fc8 = torch.nn.Linear(4096, 4096)
        self.relu8 = torch.nn.ReLU()
        self.dropout8 = torch.nn.Dropout(dropout)
        self.fc9 = torch.nn.Linear(4096, self.outputs_shape)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.inputs_shape)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.dropout5(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        x = self.fc8(x)
        x = self.relu8(x)
        x = self.dropout8(x)
        x = self.fc9(x)
        x = x.view(-1, self.depth_dim, self.features_dim, self.temporal_dim)
        return x

class LargeFCModel(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, dropout: float):
        super().__init__()
        self.depth_dim = input_shape[0]
        self.features_dim = input_shape[1]
        self.temporal_dim = input_shape[2]
        self.inputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.outputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.fc1 = torch.nn.Linear(self.inputs_shape, 4096)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout)
        self.fc3 = torch.nn.Linear(4096, 4096)
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(dropout)
        self.fc4 = torch.nn.Linear(4096, 4096)
        self.relu4 = torch.nn.ReLU()
        self.dropout4 = torch.nn.Dropout(dropout)
        self.fc5 = torch.nn.Linear(4096, 4096)
        self.relu5 = torch.nn.ReLU()
        self.dropout5 = torch.nn.Dropout(dropout)
        self.fc6 = torch.nn.Linear(4096, 4096)
        self.relu6 = torch.nn.ReLU()
        self.dropout6 = torch.nn.Dropout(dropout)
        self.fc7 = torch.nn.Linear(4096, 4096)
        self.relu7 = torch.nn.ReLU()
        self.dropout7 = torch.nn.Dropout(dropout)
        self.fc8 = torch.nn.Linear(4096, 4096)
        self.relu8 = torch.nn.ReLU()
        self.dropout8 = torch.nn.Dropout(dropout)
        self.fc9 = torch.nn.Linear(4096, 4096)
        self.relu9 = torch.nn.ReLU()
        self.dropout9 = torch.nn.Dropout(dropout)
        self.fc10 = torch.nn.Linear(4096, 4096)
        self.relu10 = torch.nn.ReLU()
        self.dropout10 = torch.nn.Dropout(dropout)
        self.fc11 = torch.nn.Linear(4096, 4096)
        self.relu11 = torch.nn.ReLU()
        self.dropout11 = torch.nn.Dropout(dropout)
        self.fc12 = torch.nn.Linear(4096, 4096)
        self.relu12 = torch.nn.ReLU()
        self.dropout12 = torch.nn.Dropout(dropout)
        self.fc13 = torch.nn.Linear(4096, 4096)
        self.relu13 = torch.nn.ReLU()
        self.dropout13 = torch.nn.Dropout(dropout)
        self.fc14 = torch.nn.Linear(4096, 4096)
        self.relu14 = torch.nn.ReLU()
        self.dropout14 = torch.nn.Dropout(dropout)
        self.fc15 = torch.nn.Linear(4096, 4096)
        self.relu15 = torch.nn.ReLU()
        self.dropout15 = torch.nn.Dropout(dropout)
        self.fc16 = torch.nn.Linear(4096, 4096)
        self.relu16 = torch.nn.ReLU()
        self.dropout16 = torch.nn.Dropout(dropout)
        self.fc17 = torch.nn.Linear(4096, self.outputs_shape)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.inputs_shape)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.dropout5(x)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        x = self.fc8(x)
        x = self.relu8(x)
        x = self.dropout8(x)
        x = self.fc9(x)
        x = self.relu9(x)
        x = self.dropout9(x)
        x = self.fc10(x)
        x = self.relu10(x)
        x = self.dropout10(x)
        x = self.fc11(x)
        x = self.relu11(x)
        x = self.dropout11(x)
        x = self.fc12(x)
        x = self.relu12(x)
        x = self.dropout12(x)
        x = self.fc13(x)
        x = self.relu13(x)
        x = self.dropout13(x)
        x = self.fc14(x)
        x = self.relu14(x)
        x = self.dropout14(x)
        x = self.fc15(x)
        x = self.relu15(x)
        x = self.dropout15(x)
        x = self.fc16(x)
        x = self.relu16(x)
        x = self.dropout16(x)
        x = self.fc17(x)
        x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)

        return x
    
class DeepLSTMModel(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, dropout: float):
        super().__init__()
        self.depth_dim = input_shape[1]
        self.features_dim = input_shape[2]
        self.temporal_dim = input_shape[0]
        self.inputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.outputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.lstm = torch.nn.RNN(self.features_dim*self.depth_dim, hidden_size = 512, batch_first=True, dropout=dropout, num_layers=16)
        self.fc1 = torch.nn.Linear(512, self.features_dim*self.depth_dim)


    def forward(self, x: torch.Tensor):
        #x = x.view(-1, self.inputs_shape)
        #x = x.view(self.temporal_dim, self.depth_dim, self.features_dim)
        x = x.view(-1, self.temporal_dim, self.features_dim* self.depth_dim)
        x, _ = self.lstm(x)
        #x = x.view(-1, 2048)
        x = self.fc1(x)
        x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        #x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        return x

class ShallowLSTMModel(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, dropout: float):
        super().__init__()
        self.depth_dim = input_shape[1]
        self.features_dim = input_shape[2]
        self.temporal_dim = input_shape[0]
        self.inputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.outputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.lstm = torch.nn.RNN(self.features_dim*self.depth_dim, hidden_size = 512, batch_first=True, dropout=dropout, num_layers=12)
        self.fc1 = torch.nn.Linear(512, self.features_dim*self.depth_dim)


    def forward(self, x: torch.Tensor):
        #x = x.view(-1, self.inputs_shape)
        #x = x.view(self.temporal_dim, self.depth_dim, self.features_dim)
        x = x.view(-1, self.temporal_dim, self.features_dim* self.depth_dim)
        x, _ = self.lstm(x)
        #x = x.view(-1, 2048)
        x = self.fc1(x)
        x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        #x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        return x

class TinyLSTMModel(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, dropout: float):
        super().__init__()
        self.depth_dim = input_shape[1]
        self.features_dim = input_shape[2]
        self.temporal_dim = input_shape[0]
        self.inputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.outputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.lstm = torch.nn.RNN(self.features_dim*self.depth_dim, hidden_size = 256, batch_first=True, dropout=dropout, num_layers=8)
        self.fc1 = torch.nn.Linear(256, self.features_dim*self.depth_dim)


    def forward(self, x: torch.Tensor):
        #x = x.view(-1, self.inputs_shape)
        #x = x.view(self.temporal_dim, self.depth_dim, self.features_dim)
        x = x.view(-1, self.temporal_dim, self.features_dim* self.depth_dim)
        x, _ = self.lstm(x)
        #x = x.view(-1, 2048)
        x = self.fc1(x)
        x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        #x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        return x
    
class TinyTransformerModel(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, dropout: float):
        super().__init__()
        self.depth_dim = input_shape[1]
        self.features_dim = input_shape[2]
        self.temporal_dim = input_shape[0]
        self.inputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.outputs_shape = self.depth_dim * self.features_dim * self.temporal_dim
        self.positional_encoder = Summer(PositionalEncoding1D(self.features_dim*self.depth_dim))
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.features_dim*self.depth_dim, nhead=8, dim_feedforward=6400, dropout=dropout, batch_first=True)
        self.layer_norm = torch.nn.LayerNorm(self.features_dim*self.depth_dim)
        self.transformer = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=8, norm=self.layer_norm)


    def forward(self, x: torch.Tensor):
        #x = x.view(-1, self.inputs_shape)
        #x = x.view(self.temporal_dim, self.depth_dim, self.features_dim)
        x = x.view(-1, self.temporal_dim, self.features_dim* self.depth_dim)
        x = self.positional_encoder(x)
        #print(f"Shape after positional encoding: {x.shape}")
        x = self.transformer(x)
        #x = x.view(-1, 2048)
        #print(f"Shape after transformer: {x.shape}")
        #x = self.fc1(x)
        x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        #x = x.view(-1, self.temporal_dim, self.depth_dim, self.features_dim)
        return x