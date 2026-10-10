# src/model.py
import torch
import torch.nn as nn
import deepxde as dde

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# -------------------
# PINN
# -------------------
class SimplePINN1(nn.Module):
    """Rede neural PINN para a equação de Burgers."""
    
    def __init__(self, input=2, layers=3, hidden=64, output=1):
        super().__init__()

        activation = nn.Tanh
        
        # Entrada: (x, t) -> 2 dimensões
        layers_list = [nn.Linear(input, hidden), activation()]
        
        # Camadas ocultas
        for _ in range(layers - 1):
            layers_list.extend([nn.Linear(hidden, hidden), activation()])
        
        # Saida: u(x,t) -> 1 dimensão
        layers_list.append(nn.Linear(hidden, output))
        
        self.net = nn.Sequential(*layers_list)
        
        # Inicialização Xavier para melhor convergência
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t):
        # Concatena x e t como entrada
        return self.net(torch.cat([x, t], dim=1))

class SimplePINN2(nn.Module):
    def __init__(self, layers, activation=nn.Tanh):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            in_dim, out_dim = layers[i], layers[i + 1]
            modules.append(nn.Linear(in_dim, out_dim))
            # ativacao apenas entre camadas (nao na saida)
            if i < len(layers) - 2:
                modules.append(activation())
        self.net = nn.Sequential(*modules)

    def forward(self, *inputs):
        # aceita x, t separados ou um tensor unico
        if len(inputs) == 1:
            x = inputs[0]
        else:
            x = torch.cat(inputs, dim=-1)
        return self.net(x)

def build_model(cfg):
    m = cfg["model"]
    name = m["name"].lower()
    if name == "simple_pinn1":
        model_config = m['layers']
        return SimplePINN1(
            input=model_config[0],
            layers=model_config[1],
            hidden=model_config[2],
            output=model_config[3],
        )
    if name == "simple_pinn2":
            layers = m['layers']
            return SimplePINN2(layers=layers)
    
    if name == "ffn":
        return dde.nn.FNN(
            m['layers'],
            m['activation'],
            m['initializer'],
        )
    raise ValueError(f"Modelo desconhecido: {name}")