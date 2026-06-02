import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

# Importo l'architettura per poter ricostruire la rete
from model import model

def calcola_laplaciano(psi: torch.Tensor, r_batch: torch.Tensor) -> torch.Tensor:
    gradiente_psi = torch.autograd.grad(
        outputs=psi, inputs=r_batch, grad_outputs=torch.ones_like(psi), 
        create_graph=True, retain_graph=True)[0] 
    
    laplaciano_psi = torch.zeros_like(psi)
    for i in range(3): 
        derivata_seconda = torch.autograd.grad(
            outputs=gradiente_psi[:, i], inputs=r_batch, grad_outputs=torch.ones_like(gradiente_psi[:, i]), 
            create_graph=True, retain_graph=True)[0]
        laplaciano_psi += derivata_seconda[:, i]
    return laplaciano_psi

def calcola_potenziale(r_batch: torch.Tensor, R_val: float, epsilon: float = 0.005) -> torch.Tensor:
    nucleo_sx = torch.zeros_like(r_batch)
    nucleo_dx = torch.zeros_like(r_batch)
    nucleo_sx[:, 0] = -R_val
    nucleo_dx[:, 0] = R_val
    
    dist_nucleo_sx = torch.norm(r_batch - nucleo_sx, dim=-1)
    dist_nucleo_dx = torch.norm(r_batch - nucleo_dx, dim=-1)
    
    V = -1.0 / torch.sqrt(dist_nucleo_sx**2 + epsilon**2) - 1.0 / torch.sqrt(dist_nucleo_dx**2 + epsilon**2)
    return V

try:
    from scipy.integrate import simpson as _simpson
except ImportError:
    from scipy.integrate import simps as _simpson

def integra3d(x, y, z, f):
    f_np = f.detach().cpu().numpy() if torch.is_tensor(f) else np.asarray(f)
    x_np = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    z_np = z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
    I = _simpson(_simpson(_simpson(f_np, x=x_np, axis=0), x=y_np, axis=0), x=z_np, axis=0)
    return float(I)

def calcola_valore_atteso_H(model, R_val, n_grid=40, box=10.0):
    coord = torch.linspace(-box, box, n_grid)
    xg, yg, zg = torch.meshgrid(coord, coord, coord, indexing='ij')
    pts = torch.stack([xg.flatten(), yg.flatten(), zg.flatten()], dim=-1)
    
    # requires_grad_(True) serve per calcolare il Laplaciano spaziale
    pts.requires_grad_(True) 
    Rt  = torch.full((pts.shape[0], 1), R_val, dtype=torch.float32)

    # PINN 
    psi_pinn, _ = model(pts, Rt)
    lap_pinn = calcola_laplaciano(psi_pinn, pts)
    V = calcola_potenziale(pts, R_val)
    
    H_psi_pinn = -0.5 * lap_pinn + V * psi_pinn
    int_num_pinn = (psi_pinn * H_psi_pinn).detach().reshape(n_grid, n_grid, n_grid)
    int_den_pinn = (psi_pinn**2).detach().reshape(n_grid, n_grid, n_grid)

    E_int_pinn = integra3d(coord, coord, coord, int_num_pinn) / integra3d(coord, coord, coord, int_den_pinn)

    # LCAO
    psi_lcao = model.calcola_LCAO(pts, Rt)
    lap_lcao = calcola_laplaciano(psi_lcao, pts)
    
    H_psi_lcao = -0.5 * lap_lcao + V * psi_lcao
    int_num_lcao = (psi_lcao * H_psi_lcao).detach().reshape(n_grid, n_grid, n_grid)
    int_den_lcao = (psi_lcao**2).detach().reshape(n_grid, n_grid, n_grid)
    
    E_int_lcao = integra3d(coord, coord, coord, int_num_lcao) / integra3d(coord, coord, coord, int_den_lcao)

    return E_int_pinn, E_int_lcao

def calcola_costante_norm_3D(model, R_val, n_grid=60, box=10.0):
    coord = torch.linspace(-box, box, n_grid)
    xg, yg, zg = torch.meshgrid(coord, coord, coord, indexing='ij')
    pts = torch.stack([xg.flatten(), yg.flatten(), zg.flatten()], dim=-1)
    Rt  = torch.full((pts.shape[0], 1), R_val, dtype=torch.float32)
    with torch.no_grad():
        psi_, _    = model(pts, Rt)
        psi_lcao_  = model.calcola_LCAO(pts, Rt)
    psi      = psi_.reshape(n_grid, n_grid, n_grid).numpy()
    psi_lcao = psi_lcao_.reshape(n_grid, n_grid, n_grid).numpy()
    coord_np = coord.numpy()
    vol_pinn = integra3d(coord, coord, coord, psi**2)
    vol_lcao = integra3d(coord, coord, coord, psi_lcao**2)
    if vol_pinn < 1e-10: vol_pinn = 1.0
    if vol_lcao < 1e-10: vol_lcao = 1.0
    return 1.0 / np.sqrt(vol_pinn), 1.0 / np.sqrt(vol_lcao)

def estrai_slicing_denso_1D(model, R_val, norm_psi, norm_lcao, n_punti=1000, box=10.0):
    asse_x = torch.linspace(-box, box, n_punti)
    pts_1D = torch.zeros(n_punti, 3)
    pts_1D[:, 0] = asse_x
    Rt_1D = torch.full((n_punti, 1), R_val, dtype=torch.float32)
    with torch.no_grad():
        psi_, _   = model(pts_1D, Rt_1D)
        psi_lcao_ = model.calcola_LCAO(pts_1D, Rt_1D)
    return asse_x.numpy(), (psi_ * norm_psi).numpy(), (psi_lcao_ * norm_lcao).numpy()
    
if __name__ == '__main__':
    model = model(input_dim=4, n_hidden=2, n_neurons=16)
    
    checkpoint = torch.load('best_model_results.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    valori_loss_PDE = checkpoint['valori_loss_PDE']
    valori_loss_BC = checkpoint['valori_loss_BC']
    valori_loss = checkpoint['valori_loss']
    epoche_pre_ft = checkpoint.get('epoche_pre_ft', 4000)

    # GRAFICO 1: Loss
    plt.figure(figsize=(10, 6))
    plt.plot(valori_loss_PDE, color='red', linewidth=2, label='Loss PDE')
    plt.plot(valori_loss_BC, color='green', linewidth=2, label='Loss BC')
    plt.plot(valori_loss, color='blue', linewidth=4, alpha=0.3, label='Total Loss')
    plt.axvline(x=epoche_pre_ft, color='k', linestyle='--', label='Inizio Fine-Tuning')
    
    plt.xlabel('Epoca')
    plt.ylabel('Valore Loss')
    plt.grid(True, alpha=0.3)
    plt.title('Andamento delle componenti della Loss')
    plt.yscale('log') 
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calcolo valori di aspettazione <H>
    
    rE = np.arange(0.5, 4.1, 0.5) # Punti R scelti per l'integrazione (stile Paper)
    E_int_list = []
    E_int_lcao_list = []

    model.train() # Serve per l'autograd nel Laplaciano
    for r_val in rE:
        e_pinn, e_lcao = calcola_valore_atteso_H(model, r_val, n_grid=40, box=10.0)
        E_int_list.append(e_pinn)
        E_int_lcao_list.append(e_lcao)

    E_int = np.array(E_int_list)
    E_int_lcao = np.array(E_int_lcao_list)
    
    # Aggiungiamo la repulsione nucleare 1 / (2*R)
    E_tot_int = E_int + 1.0 / (2.0 * rE)
    E_tot_int_lcao = E_int_lcao + 1.0 / (2.0 * rE)

    # Dati Neurali Continui (E(R) del modello) 
    R_forza = torch.linspace(0.25, 4.0, 200, requires_grad=True).unsqueeze(1)
    E_el_neural = model.energy_unit(R_forza).squeeze(-1)
    repulsione = 1.0 / (2.0 * R_forza.squeeze(-1))
    E_tot_neural = E_el_neural + repulsione
    
    # Forza Autograd
    Forza_PINN = -torch.autograd.grad(
        outputs=E_tot_neural, inputs=R_forza, grad_outputs=torch.ones_like(E_tot_neural), create_graph=False
    )[0].squeeze(-1).numpy()
    
    model.eval()
    R_np = R_forza.detach().squeeze(-1).numpy()
    E_tot_neural_np = E_tot_neural.detach().numpy()
    
    with torch.no_grad():
        gate_values = model.gate(R_forza).squeeze(-1).numpy()

    # Dati Reference Esatti (Wind) 
    Rexact = np.round(np.arange(0.2, 4.1, 0.1), 2)  
    E_elec_exact = np.array([
        -1.8008, -1.6715, -1.5545, -1.4518, -1.3623, -1.2843, -1.2159, -1.1558,
        -1.1026, -1.0554, -1.0132, -0.9754, -0.9415, -0.9109, -0.8832, -0.8582, 
        -0.8355, -0.8149, -0.7961, -0.7790, -0.7634, -0.7492, -0.7363, -0.7244, 
        -0.7136, -0.7037, -0.6946, -0.6863, -0.6786, -0.6716, -0.6651, -0.6591, 
        -0.6536, -0.6485, -0.6437, -0.6392, -0.6351, -0.6312, -0.6276
    ])
    e_exact_tot = E_elec_exact + (1.0 / (2.0 * Rexact))
    F_ex = -np.gradient(e_exact_tot, Rexact)

    # Forze alle differenze finite sugli integrali
    F_int = -np.gradient(E_tot_int, rE)
    F_lcao = -np.gradient(E_tot_int_lcao, rE)

    # Errori Interpolati
    e_exact_rE = np.interp(rE, Rexact, e_exact_tot)
    df_int = E_tot_int - e_exact_rE
    df_lcao = E_tot_int_lcao - e_exact_rE
    
    # Errore continuo per E(R) neurale
    with torch.no_grad():
        R_ex_tens = torch.tensor(Rexact, dtype=torch.float32).unsqueeze(1)
        E_net_grid = model.energy_unit(R_ex_tens).squeeze(-1).numpy() + (1.0/(2.0*Rexact))
    df_net = E_net_grid - e_exact_tot


    # GRAFICO COMPOSITO: Energy, Force, Errori, Gate

    lineW = 3
    marker_style_exact = dict(color='r', linestyle='none', marker='o', fillstyle='full', linewidth=lineW)
    marker_style_int = dict(color='g', linestyle='none', marker='o', fillstyle='none', linewidth=lineW)
    marker_style_net = dict(color='b', linestyle='-', linewidth=lineW)
    marker_style_lcao = dict(color='m', linestyle='none', marker='*', fillstyle='none', linewidth=lineW)

    fig = plt.figure(figsize=(20, 10))

    # (in alto a sinistra) Potential Energy Surface
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1, rowspan=2)
    plt.tick_params('x', labelbottom=False)

    plt.plot(Rexact, e_exact_tot, **marker_style_exact, label='Reference: $E_{t}$')
    plt.plot(rE, E_tot_int_lcao, **marker_style_lcao, label='LCAO: $\langle E_\ell \\rangle$')
    plt.plot(rE, E_tot_int, **marker_style_int, label ='Neural: $\langle \hat H \\rangle$')
    plt.plot(R_np, E_tot_neural_np, **marker_style_net, label='Neural: $E(R)$')
    
    plt.legend(frameon=False)
    plt.xlim([0.25, 4.0]); plt.ylim([-0.65, 0]); plt.ylabel('Energy (AU)')
    plt.grid(True, alpha=0.3)

    # (in basso a sinistra) Errore
    plt.subplot2grid((3, 2), (2, 0), colspan=1, sharex=ax1)
    plt.tick_params('x', labelbottom=True)
    
    plt.plot(rE, df_int, **marker_style_int)
    plt.plot(rE, df_lcao, **marker_style_lcao)
    plt.plot(Rexact, df_net, **marker_style_net)
    
    plt.ylabel('Error (AU)'); plt.ylim(-0.02, 0.06); plt.yticks([0, 0.025, 0.05])
    plt.axhline(0, c='k', linestyle='--', linewidth=lineW*1.0, alpha=0.9)
    plt.xlabel("$R$"); plt.xticks(np.arange(0.5, 4.5, 0.5))
    plt.grid(True, alpha=0.3)

    # (in alto a destra) Forza
    ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1, rowspan=2)
    plt.tick_params('x', labelbottom=False)

    plt.plot(Rexact[1:-1], F_ex[1:-1], **marker_style_exact)
    plt.plot(rE[1:-1], F_lcao[1:-1], **marker_style_lcao)
    plt.plot(rE[1:-1], F_int[1:-1], **marker_style_int)
    plt.plot(R_np, Forza_PINN, '--b', linewidth=lineW)

    plt.ylabel('Force'); plt.ylim([-0.1, 0.5]); plt.xlim([0.25, 4.0])
    plt.legend(frameon=False)
    plt.axhline(0, c='k', linestyle='--', linewidth=lineW*1, alpha=0.9)
    plt.grid(True, alpha=0.3)

    # (in basso a destra) Gate
    plt.subplot2grid((3, 2), (2, 1), colspan=1, sharex=ax2)
    plt.plot(R_np, gate_values, 'c', linewidth=lineW)
    plt.ylabel('Gate'); plt.xlabel("$R$"); plt.xticks(np.arange(0.5, 4.5, 0.5))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # GRAFICI 1D WAVEFUNCTIONS (R=1.0 e R=2.0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for R_val, ax, color in [(1.0, ax1, 'blue'), (2.0, ax2, 'purple')]:
        N_pinn, N_lcao = calcola_costante_norm_3D(model, R_val=R_val, n_grid=60, box=10.0)
        x_axis, psi_cut, lcao_cut = estrai_slicing_denso_1D(model, R_val=R_val, norm_psi=N_pinn, norm_lcao=N_lcao)

        ax.plot(x_axis, psi_cut,  color=color, linewidth=2.5, label='Psi PINN (norm 3D)')
        ax.plot(x_axis, lcao_cut, color='gray', linestyle='--', linewidth=2, label='Psi LCAO (norm 3D)')
        ax.scatter([-R_val, R_val], [0, 0], color='red', s=5, zorder=5, label='Nuclei')
        #ax.axvline(x=-R_val, color='red', linestyle=':', alpha=0.5)
        #ax.axvline(x=+R_val, color='red', linestyle=':', alpha=0.5)
        
        ax.set_title(f"Funzione d'onda lungo l'asse internucleare (R = {R_val} a.u.)")
        ax.set_ylabel('Ampiezza normalizzata')
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax2.set_xlabel('x [a.u.]')
    plt.tight_layout()
    plt.show()

    # GRAFICI 3D SURFACE WAVEFUNCTIONS (R=1.0 e R=2.0)
    
    def calcola_costante_norm_3D_base(model, R_val, n_grid=60, box=10.0):
        coord = torch.linspace(-box, box, n_grid)
        xg, yg, zg = torch.meshgrid(coord, coord, coord, indexing='ij')
        pts = torch.stack([xg.flatten(), yg.flatten(), zg.flatten()], dim=-1)
        Rt  = torch.full((pts.shape[0], 1), R_val, dtype=torch.float32)

        with torch.no_grad():
            psi_, _ = model(pts, Rt)
        
        psi = psi_.reshape(n_grid, n_grid, n_grid).numpy()
        vol_pinn = integra3d(coord, coord, coord, psi**2)
        return 1.0 / np.sqrt(max(vol_pinn, 1e-10))

    n_grid_surf = 100  
    box_surf = 5.0     
    
    fig_surf = plt.figure(figsize=(20, 10)) 
    valori_R = [1.0, 2.0]
    
    for i, R_val_surf in enumerate(valori_R):
        
        N_pinn_surf = calcola_costante_norm_3D_base(model, R_val=R_val_surf, n_grid=60, box=10.0)

        coord_surf = torch.linspace(-box_surf, box_surf, n_grid_surf)
        xg_surf, yg_surf = torch.meshgrid(coord_surf, coord_surf, indexing='ij')
        zg_surf = torch.zeros_like(xg_surf)

        pts_surf = torch.stack([xg_surf.flatten(), yg_surf.flatten(), zg_surf.flatten()], dim=-1)
        Rt_surf  = torch.full((pts_surf.shape[0], 1), R_val_surf, dtype=torch.float32)

        with torch.no_grad():
            psi_surf_, _ = model(pts_surf, Rt_surf)
        
        psi_surf_n = (psi_surf_ * N_pinn_surf).reshape(n_grid_surf, n_grid_surf).numpy()
        xg_np = xg_surf.numpy()
        yg_np = yg_surf.numpy()

        ax_surf = fig_surf.add_subplot(1, 2, i + 1, projection='3d')
        surf = ax_surf.plot_surface(xg_np, yg_np, psi_surf_n, cmap=cm.coolwarm, 
                                    antialiased=True, linewidth=0, shade=False)

        ax_surf.set_title(f"Wavefunction Surface (R = {R_val_surf} a.u.)", fontsize=16)
        ax_surf.set_xlabel("$x$")
        ax_surf.set_ylabel("$y$")
        ax_surf.set_xticks([])
        ax_surf.set_yticks([])
        ax_surf.set_zticks([])
        ax_surf.view_init(elev=30, azim=60)
        ax_surf.grid(False)
        ax_surf.axis('off')

    plt.tight_layout()
    plt.show()