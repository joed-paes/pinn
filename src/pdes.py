# src/pdes.py
#import deepxde as dde
import numpy as np
import torch

PDE_REGISTRY = {}

def register(name):
    def wrapper(fn):
        if name in PDE_REGISTRY:
            raise ValueError(f"PDE '{name}' ja registrada")
        PDE_REGISTRY[name] = fn
        return fn
    return wrapper

def compute_grad(outputs, inputs, create_graph=True, retain_graph=True):
    """Funcao utilitaria para calcular derivadas no PyTorch de forma limpa."""
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=retain_graph
    )[0]

#@register("heat_1d")
#def heat_1d(coords, u, params):
#    """∂u/∂t = α ∂²u/∂x²  →  resíduo: u_t - α u_xx"""
#    x = coords
#    alpha = params.get("alpha", 0.01)
#    du_t = dde.grad.jacobian(u, x, i=0, j=1)
#    du_xx = dde.grad.hessian(u, x, i=0, j=0)
#    return du_t - alpha * du_xx


#@register("poisson_2d")
#def poisson_2d(coords, u, params):
#    """-∇²u = f  →  resíduo: -u_xx - u_yy - f"""
#    x = coords
#    f = params.get("f", 1.0)
#    du_xx = dde.grad.hessian(u, x, i=0, j=0)
#    du_yy = dde.grad.hessian(u, x, i=1, j=1)
#    return -du_xx - du_yy - f

@register("burger")
def builder_burger(params):
    """
    Equacao de Burgers (Viscosa (nu != 0) ou Inviscida (nu==0)).
    coords: tupla ou lista contendo (x, t)
    Residuo: u_t + u * u_x - nu * u_xx = 0
    """
    bc_info = params["boundary"]
    ic_type = params.get("ic_type", "sin")
    nu = params.get("nu", 0.0)
    n_x = params.get("n_x", 1.0)    # size of spatial domain (1D, n_x=1; 2D, n_x=2; ...)
    def residual(coords, u):
        #x, t = torch.split(coords, split_size_or_sections=[n_x, 1], dim=1)
        x, t = coords
        u_t = compute_grad(u, t)
        u_x = compute_grad(u, x)
        residual = u_t + u*u_x

        advection = u*u_x

        if nu > 0:
            # Derivada segunda espacial - termo viscoso
            u_xx = compute_grad(u_x, x)
            residual = residual - nu*u_xx

        return residual

    # Function of initial condition
    # As t=0, it receives only x
    def ic_true_fn(x_coords):
        functions = {
            "sin": lambda x: -torch.sin(torch.pi*x),
            "tanh": lambda x: 1 - 0.8*torch.tanh(2*x),  
        }
        u0 = functions.get(ic_type.lower(), functions["sin"])
        return u0(x_coords)

    # Function derivative of initial condition
    def ic_true_fn_prime(x_coords):
        functions = {
            "sin": lambda x: -torch.cos(torch.pi*x),
            "tanh": lambda x: (8*torch.tanh(2*x)**2)/5 - 8/5,  
        }
        u0_prime = functions.get(ic_type.lower(), functions["sin"])
        return u0_prime(x_coords)

    # Boundary conditions
    # Receive the coordinates and return the expected value
    def bc_true_fn(x_coords, t_coords):
        n_half = len(x_coords) // 2
        functions = {
            "sin": lambda n: torch.cat([torch.full((n,1), bc_info["u_left"]), torch.full((n,1), bc_info["u_right"])], dim=0),   # BCs: u(-x_min, t)=0.0, u(x_max,t)=0.0
            "tanh": lambda n: torch.cat([torch.full((n,1), bc_info["u_left"]), torch.full((n,1), bc_info["u_right"])], dim=0),  # BCs: u(-x_min, t)=1.8, u(x_max,t)=0.2
        }
        bc = functions.get(ic_type.lower(), functions["sin"])
        return bc(n_half)

    # Include Jacobian function in order to determine caustic geometry
    # Why here: because Jacobian in intrinsec from PDE itself
    # def jacobian(x_coords, t_coords)

    def jacobian(x_coords, t_coords):
        """
        Jacobiano das características: J(x0,t) = 1 + t*u0'(x0)
        O Jacobiano se anula na formação do choque.
        """
        return 1.0 + t_coords * ic_true_fn_prime(x_coords)
    
    return residual, ic_true_fn, bc_true_fn


def build_pde(cfg):
    """Retorna a funcao de residuo da EDP e o dict de condicoes."""
    name = cfg["pde"]["name"]
    if name not in PDE_REGISTRY:
        raise ValueError(f"EDP desconhecida: {name}")
    params = cfg["pde"].get("params", {})
    # Extract PDE specific functions
    residual_fn, ic_fn, bc_fn = PDE_REGISTRY[name](params)
    # Encapsulates the residual
    pde_residual_fn = lambda coords, u: residual_fn(coords, u)
    return pde_residual_fn, ic_fn, bc_fn