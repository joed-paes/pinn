# src/geometry.py
import torch
import numpy as np
from scipy.stats import qmc

class SpatialTemporalDomain:
    def __init__(self, cfg):
        """
        Gerencia o domínio espaço-temporal e a amostragem de pontos.
        """
        self.device = cfg["device"]
        geom_cfg = cfg["pde"]["params"]["domain"]
        self.sampling = cfg["sampling"]
        self.x_min, self.x_max = geom_cfg["x_min"], geom_cfg["x_max"] 
        self.t_min, self.t_max = geom_cfg["t_min"], geom_cfg["t_max"]
        
        # Tipo de amostragem vindo da configuração (ex: 'uniform', 'latin_hypercube', 'jacobian')
        self.sampling_method = cfg.get("sampling").get("sampling_method", "uniform")

    def sample_collocation(self):
        """Amostra pontos da fisica (PDE) baseado no metodo escolhido."""
        num_points = self.sampling["num_domain"]
        if self.sampling_method == "uniform":
            # Distribuicao uniforme aleatoria padrao
            x = (self.x_max - self.x_min) * torch.rand((num_points, 1)) + self.x_min
            t = (self.t_max - self.t_min) * torch.rand((num_points, 1)) + self.t_min

        elif self.sampling_method == "latin_hypercube":
            # Amostragem por Hipercubo Latino usando o SciPy (2 dimensoes: x e t)
            sampler = qmc.LatinHypercube(d=2)
            sample = sampler.random(n=num_points) # Retorna pontos no intervalo [0, 1)
            
            # Mapeamento do intervalo [0, 1] para os limites reais [min, max]
            x_raw = sample[:, 0:1]
            t_raw = sample[:, 1:2]
            
            x = torch.tensor(self.x_min + (self.x_max - self.x_min) * x_raw, dtype=torch.float32)
            t = torch.tensor(self.t_min + (self.t_max - self.t_min) * t_raw, dtype=torch.float32)

        elif self.sampling_method == "caustic_geometric":
            # Concentra os pontos onde a descontinuidade (choque) da Equação de Burgers ocorre
            # Geralmente o choque se desenvolve centralizado em x=0 conforme t avança.
            # Usamos uma transformação não-linear (ex: função arco-tangente) para adensar o centro do domínio espacial.
            u_rand = torch.rand((num_points, 1)) * 2 - 1 # Intervalo [-1, 1]
            
            # Empurra os pontos em direção ao zero (ponto caustico de Burgers)
            # Fator de concentração (ex: 3.0). Quanto maior, mais concentrado no centro.
            factor = 3.0 
            x_compressed = torch.atan(factor * u_rand) / torch.atan(torch.tensor(factor))
            
            # Remapeia de [-1, 1] para [x_min, x_max]
            x = self.x_min + (self.x_max - self.x_min) * (x_compressed + 1.0) / 2.0
            
            # O tempo pode seguir um mapeamento normal ou linearizado uniforme
            t = (self.t_max - self.t_min) * torch.rand((num_points, 1)) + self.t_min

        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        # Ativa o rastreamento de gradientes para o calculo do Autograd
        x.requires_grad_(True)
        t.requires_grad_(True)
        return x.to(self.device), t.to(self.device)

    def sample_initial(self):
        """Amostra pontos na linha temporal inicial (t = 0)."""
        num_points = self.sampling["num_initial"]
        # Geralmente mantido como uniforme ou linearmente espaçado para a condição inicial
        x = (self.x_max - self.x_min) * torch.rand((num_points, 1)) + self.x_min
        t = torch.zeros((num_points, 1)) + self.t_min
        return x.to(self.device), t.to(self.device)

    def sample_boundary(self):
        """Amostra pontos nas bordas espaciais (x = x_min e x = x_max)."""
        num_points = self.sampling["num_boundary"]
        x_left = torch.full((num_points, 1), self.x_min)
        x_right = torch.full((num_points, 1), self.x_max)
        x = torch.cat([x_left, x_right], dim=0)
        
        t = (self.t_max - self.t_min) * torch.rand((2*num_points, 1)) + self.t_min
        return x.to(self.device), t.to(self.device)
