import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinv

"""
AUTONOMOUS UNDERWATER VEHICLE (AUV) SIMULATION WITH SLIDING MODE CONTROL

This simulation demonstrates path following for an AUV using Sliding Mode Control (SMC).
SMC provides robust control performance against model uncertainties and external disturbances.

== SLIDING MODE CONTROL THEORY ==

Sliding Mode Control (SMC) is a nonlinear control method that alters the dynamics of a system
by applying a discontinuous control signal that forces the system to "slide" along a cross-section
of the system's normal behavior.

The key concept in SMC is the sliding surface (s), defined as:

    s(x, t) = error_dot + λ·error = 0

For a second-order system, where:
- error = x_d - x (desired state - current state)
- λ > 0 is a design parameter that defines the convergence rate

When the system state is on the sliding surface (s = 0), the error converges to zero
exponentially with time constant 1/λ.

The control law consists of two components:
1. Equivalent control: Keeps the system on the sliding surface
2. Switching control: Drives the system to the sliding surface

    u = u_eq + K_S·sign(s)

where:
- u_eq is often approximated as K_D·s
- K_S is the switching gain
- sign(s) is the signum function

== LYAPUNOV STABILITY ANALYSIS ==

The stability of SMC is proven using Lyapunov's direct method. The Lyapunov function is:

    V = 0.5·s^T·M·s > 0

where M is the inertia matrix.

The derivative of V is:
    
    dV/dt = s^T·M·ds/dt + 0.5·s^T·dM/dt·s
    
Using the system dynamics and the control law, we can show that:

    dV/dt = -s^T·D·s - s^T·K_D·s + s^T·(uncertainties - K_S·sign(s))

By choosing K_S large enough such that:

    min(eigenvalues(K_S)) > ||uncertainties||
    
We ensure dV/dt < 0, which guarantees the system stability according to Lyapunov theory.
"""

def wrap_to_pi(angle):
    """
    Membungkus sudut ke dalam rentang [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Inisiasi kondisi awal dan parameter kontrol
def initialize_parameters():
    # === Hull Parameters (Table 2.7) ===
    hull = {
        'rho': 1.03e3,          # kg/m³, Seawater Density
        'A_f': 2.85e-2,         # m², Hull Frontal Area
        'A_p': 2.26e-1,         # m², Hull Projected Area (xz plane)
        'S_w': 7.09e-1,         # m², Hull Wetted Surface Area
        'nabla': 3.15e-2,       # m³, Estimated Hull Volume
        'B_est': 3.17e2,        # N, Estimated Hull Buoyancy
        'x_cb_est': 5.54e-3     # m, Estimated Long. Center of Buoyancy
    }
    # === Non-Linear Maneuvering Coefficients: Forces (Table 4.2) ===
    forces = {
        'X_uu': -1.62e0,      # kg/m, Cross-flow Drag
        'X_u': -9.30e-1,      # kg, Added Mass
        'X_wq': -3.35e1,      # kg/rad, Added Mass Cross-term
        'X_qq': -1.93e0,      # kg·m/rad, Added Mass Cross-term
        'X_vr': 3.55e1,       # kg·m/rad, Added Mass Cross-term
        'X_rr': -1.93e0,      # kg·m/rad, Added Mass Cross-term
        'X_prop': 3.38e0,     # N, Propeller Thrust
        'Y_vv': -1.31e2,      # kg/m, Cross-flow Drag
        'Y_rr': 6.32e-1,      # kg·m/rad², Cross-flow Drag
        'Y_uv': -2.86e1,      # kg/m, Body Lift Force and Fin Lift
        'Y_v': -3.55e1,       # kg, Added Mass
        'Y_r': 1.93e0,        # kg·m/rad, Added Mass
        'Y_ur': 5.22e0,       # kg/rad, Added Mass Cross Term and Fin Lift
        'Y_wp': 3.55e1,       # kg/rad, Added Mass Cross-term
        'Y_pq': 1.93e0,       # kg·m/rad, Added Mass Cross-term
        'Y_udr': 9.64e0,      # kg/(m·rad), Fin Lift Force
        'Z_ww': -1.31e2,      # kg/m, Cross-flow Drag
        'Z_qq': -6.32e-1,     # kg·m/rad², Cross-flow Drag
        'Z_uw': -2.86e1,      # kg/m, Body Lift Force and Fin Lift
        'Z_w': -1.93e0,       # kg, Added Mass
        'Z_qdot': -1.93e0,    # kg·m/rad, Added Mass
        'Z_uq': -5.22e0,      # kg/rad, Added Mass Cross Term and Fin Lift
        'Z_vp': -3.55e1,      # kg/rad, Added Mass Cross-term
        'Z_rp': 1.93e0,       # kg/rad, Added Mass Cross-term
        'Z_uuds': -9.64e0     # kg/(m·rad), Fin Lift Force
    }
    # === Non-Linear Maneuvering Coefficients: Moments (Table 4.3) ===
    moments = {
        'K_pp': -1.30e-3,      # kg·m²/rad², Rolling Resistance
        'K_p': -1.41e-2,       # kg·m²/rad, Added Mass
        'K_prop': -5.43e-1,    # N·m, Propeller Torque
        'M_ww': 3.18e0,        # kg, Cross-flow Drag
        'M_qq': -9.40e1,       # kg·m²/rad², Cross-flow Drag
        'M_uw': 2.40e1,        # kg, Body and Fin Lift and Munk Moment
        'M_w': -1.93e0,        # kg·m, Added Mass
        'M_qdot': -4.88e0,     # kg·m²/rad, Added Mass
        'M_uq': -2.00e0,       # kg·m²/rad, Added Mass Cross Term and Fin Lift
        'M_vp': -1.93e0,       # kg·m²/rad, Added Mass Cross-term
        'M_rp': 4.86e0,        # kg·m²/rad², Added Mass Cross-term
        'M_uuds': -6.15e0,     # kg/rad, Fin Lift Moment
        'N_vv': -3.18e0,       # kg·m²/rad², Cross-flow Drag
        'N_rr': -9.40e1,       # kg·m²/rad², Cross-flow Drag
        'N_uw': -2.40e1,       # kg, Body and Fin Lift and Munk Moment
        'N_v': 1.93e0,         # kg·m, Added Mass
        'N_r': -4.88e0,        # kg·m²/rad, Added Mass
        'N_ur': -2.00e0,       # kg·m²/rad, Added Mass Cross Term and Fin Lift
        'N_wp': -1.93e0,       # kg·m²/rad, Added Mass Cross-term
        'N_pq': -4.86e0,       # kg·m²/rad², Added Mass Cross-term
        'N_uadr': -6.15e0      # kg/rad, Fin Lift Moment
    }
    # === Control and Simulation Parameters ===
    control = {
        # Lambda values for individual controllers:
        # [surge, sway, heave, yaw]
        'Lambda': np.diag([0.2, 0.2, 0.5, 0.3]),  # Reduced lambda values for better stability
        'KD': 20 * np.eye(2),                     # Reduced KD for smoother control (Surge and Yaw)
        'KS': 5 * np.eye(2),                      # Reduced KS to reduce chattering (Surge and Yaw)
        'K_theta': np.eye(5),                     # Placeholder for compatibility
        'theta_hat': np.zeros(5),                 # Placeholder for compatibility
        'theta_dot': np.zeros(5),                 # Placeholder for compatibility
        'B': np.eye(2),                           # Control input matrix for [surge, yaw]
        'u': np.array([0.0, 0.0])                 # Initialize control input [surge, yaw]
    }
    # === Simulation Parameters ===
    simulation = {
        'Tend': 100,                     # Simulation time
        'dt': 0.1,                      # Smaller time step for better stability
        'waypoints': np.array([[0, 0], [15, 0]]),  # Waypoints
        'switchRadius': 1.5,             # Increased radius to switch waypoint
        'lookAhead': 5.0,                # Increased lookahead distance for smoother path following
        'lam_psi': 0.5,                  # Reduced lambda for heading control
        'k_psi': 1.0,                    # Reduced gain for heading control
        'lam_u': 0.3,                    # Reduced lambda for surge control
        'k_u': 5.0,                      # Reduced gain for surge control
        'phi_sat': 0.2,                  # Increased saturation parameter for smoother control
        'u_des': 0.3,                    # Reduced desired surge velocity for stability
        'n_max': 25,                     # Max rev/s
        'kT': 0.04,                      # Propeller thrust constant
        'kM': 0.01,                      # Propeller torque constant
        'delta_max': np.deg2rad(15),     # Reduced rudder limit for stability
        'Uc': 0.05,                      # Reduced ocean current (east)
        'Vc': 0.05,                      # Reduced ocean current (north)
        'A_f': 2.85e-2,                  # m², Hull Frontal Area
        'A_p': 2.26e-1                   # m², Hull Projected Area
    }
    return hull, forces, moments, control, simulation

# 4. Implementasi LOS Guidance
def los_guidance(eta, waypoints, wpIdx, simulation):
    """
    Menghitung desired heading (psi_d) dan cross-track error (e_ct) berdasarkan LOS Guidance.
    Parameters:
        eta (np.array): [x, y, psi]
        waypoints (np.array): Array waypoints
        wpIdx (int): Indeks waypoint saat ini
        simulation (dict): Parameter simulasi
    Returns:
        psi_d (float): Desired heading
        e_ct (float): Cross-track error
        wpIdx_new (int): Updated waypoint index
    """
    if wpIdx >= len(waypoints)-1:
        # Target dicapai
        psi_d = eta[2]
        e_ct = 0.0
        return psi_d, e_ct, wpIdx
    p_i = waypoints[wpIdx]
    p_ip1 = waypoints[wpIdx + 1]
    distance_to_wp = np.linalg.norm(eta[0:2] - p_ip1)
    if distance_to_wp < simulation['switchRadius']:
        wpIdx += 1
        if wpIdx >= len(waypoints)-1:
            psi_d = eta[2]
            e_ct = 0.0
            return psi_d, e_ct, wpIdx
        p_i = waypoints[wpIdx]
        p_ip1 = waypoints[wpIdx + 1]
        distance_to_wp = np.linalg.norm(eta[0:2] - p_ip1)
    psi_path = np.arctan2(p_ip1[1] - p_i[1], p_ip1[0] - p_i[0])
    e = eta[0:2] - p_i
    e_ct = -np.sin(psi_path)*e[0] + np.cos(psi_path)*e[1]
    psi_d = psi_path - np.arctan(e_ct / simulation['lookAhead'])
    return psi_d, e_ct, wpIdx

# 5. Implementasi Sliding Mode Control
def sliding_mode_control(reference, measurement, derivative, control_params, control_type='surge'):
    """
    Menghitung input kontrol menggunakan Sliding Mode Control (SMC).
    Parameters:
        reference (float): Referensi target (contoh: desired surge velocity atau desired yaw)
        measurement (float): Pengukuran saat ini (contoh: current surge velocity atau current yaw)
        derivative (float): Derivatif kesalahan (contoh: turunan surge atau yaw)
        control_params (dict): Parameter kontrol termasuk KD, KS, dan Lambda
        control_type (str): Tipe kontrol ('surge' atau 'yaw')
    Returns:
        control_signal (float): Sinyal kontrol
        s (float): Sliding manifold
    """
    # Hitung kesalahan
    error = reference - measurement
    
    # Pilih parameter kontrol berdasarkan tipe
    if control_type == 'yaw':
        idx = 1  # Indeks untuk yaw di KD dan KS
        lambda_val = control_params['Lambda'][3, 3]  # Nilai dari diagonal ke-4 (untuk yaw)
    else:  # Default: surge
        idx = 0  # Indeks untuk surge di KD dan KS
        lambda_val = control_params['Lambda'][0, 0]  # Nilai dari diagonal pertama (untuk surge)
    
    # Definisikan sliding surface s = error_dot + lambda * error
    s = derivative + lambda_val * error
    
    # Hitung kontrol SMC dengan nilai KD dan KS yang sesuai
    KD_val = control_params['KD'][idx, idx]  # Nilai dari diagonal KD
    KS_val = control_params['KS'][idx, idx]  # Nilai dari diagonal KS
    
    # Tambahkan saturasi untuk mengurangi chattering
    phi = 0.1
    sat_term = np.sign(s) if abs(s) > phi else s/phi
    
    # Kontrol SMC dengan boundary layer untuk mengurangi chattering
    control_signal = (KD_val * s) + (KS_val * sat_term)
    
    # Kontrol sinyal dibatasi untuk mencegah kontrol yang terlalu besar
    max_value = 50.0 if control_type == 'surge' else 20.0  # Batasan yang berbeda untuk surge dan yaw
    control_signal = np.clip(control_signal, -max_value, max_value)
    
    return control_signal, s

# 5.1 Analisis Stabilitas Lyapunov untuk SMC
def lyapunov_stability_analysis(M, s, D, KD, KS, error_bound):
    """
    Melakukan analisis stabilitas Lyapunov untuk Sliding Mode Control
    Parameters:
        M: Matriks inersia (massa)
        s: Sliding surface
        D: Matriks damping
        KD: Matriks penguatan definit positif
        KS: Matriks penguatan signum
        error_bound: Batas atas kesalahan model dan gangguan
    Returns:
        is_stable (bool): Apakah sistem stabil menurut analisis Lyapunov
        V: Nilai fungsi Lyapunov
        V_dot: Nilai turunan fungsi Lyapunov
    """
    # Fungsi Lyapunov: V = 0.5 * s^T * M * s
    V = 0.5 * s.T @ M @ s
    
    # Kondisi stabilitas: λ_min(KS) > ||error_bound||
    min_eigenvalue_KS = np.min(np.diag(KS))
    
    # Turunan fungsi Lyapunov: V_dot = -s^T(D+KD)s + s^T(error_bound - KS*sign(s))
    V_dot_stability = -s.T @ (D + KD) @ s
    
    # Menggunakan saturasi untuk menghindari chattering
    phi = 0.1
    sat_term = np.sign(s) if abs(s) > phi else s/phi
    V_dot_robustness = s * (error_bound - KS * sat_term)
    
    V_dot = V_dot_stability + V_dot_robustness
    
    # Sistem stabil jika V_dot < 0 dan min_eigenvalue_KS > error_bound
    is_stable = (V_dot < 0) and (min_eigenvalue_KS > error_bound)
    
    return is_stable, V, V_dot

# 5.2 Visualisasi dan Analisis Kestabilan SMC
def visualize_sliding_mode_stability(t_vec, s_surge, s_yaw, eta, nu, control_u, control_r):
    """
    Memvisualisasikan dan menganalisis kestabilan Sliding Mode Control
    Parameters:
        t_vec: Vektor waktu
        s_surge: Sliding surface untuk surge
        s_yaw: Sliding surface untuk yaw
        eta: [x, y, psi] - posisi dan orientasi
        nu: [u, v, r] - kecepatan
        control_u: Kontrol surge
        control_r: Kontrol yaw
    """
    # Plotting Phase Portrait untuk Sliding Mode (error vs error_dot)
    fig_phase, ax_phase = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Phase Portrait untuk Surge
    # Hitung error dan error_dot dari sliding surface: s = error_dot + lambda*error
    # Misal lambda = 0.2 (dari parameter kontrol)
    lambda_surge = 0.2
    surge_error = nu[:, 0] - 0.3  # Error (u - u_des), u_des = 0.3
    surge_error_dot = s_surge - lambda_surge * surge_error
    
    ax_phase[0].plot(surge_error, surge_error_dot, 'b-', label='Surge Trajectory')
    ax_phase[0].plot(0, 0, 'ro', label='Desired State')
    # Plot sliding surface: error_dot + lambda*error = 0
    x_range = np.linspace(-0.5, 0.5, 100)
    ax_phase[0].plot(x_range, -lambda_surge * x_range, 'k--', label=f'Sliding Surface (λ={lambda_surge})')
    ax_phase[0].set_title('Phase Portrait: Surge Control')
    ax_phase[0].set_xlabel('Error')
    ax_phase[0].set_ylabel('Error Rate')
    ax_phase[0].grid(True)
    ax_phase[0].legend()
    
    # 2. Phase Portrait untuk Yaw
    # Misal lambda = 0.3 (dari parameter kontrol)
    lambda_yaw = 0.3
    # Assuming heading error is stored in eta[:, 2] (orientation error)
    yaw_error = np.radians(np.diff(np.concatenate(([0], eta[:, 2]))))
    yaw_error_dot = s_yaw[1:] - lambda_yaw * yaw_error
    
    ax_phase[1].plot(yaw_error, yaw_error_dot, 'r-', label='Yaw Trajectory')
    ax_phase[1].plot(0, 0, 'bo', label='Desired State')
    # Plot sliding surface: error_dot + lambda*error = 0
    x_range = np.linspace(-0.5, 0.5, 100)
    ax_phase[1].plot(x_range, -lambda_yaw * x_range, 'k--', label=f'Sliding Surface (λ={lambda_yaw})')
    ax_phase[1].set_title('Phase Portrait: Yaw Control')
    ax_phase[1].set_xlabel('Error')
    ax_phase[1].set_ylabel('Error Rate')
    ax_phase[1].grid(True)
    ax_phase[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 3. Lyapunov Stability Analysis
    fig_lyap, ax_lyap = plt.subplots(2, 1, figsize=(14, 10))
    
    # Estimate Lyapunov function V = 0.5 * s^T * s (simplified version)
    V_surge = 0.5 * s_surge**2
    V_yaw = 0.5 * s_yaw**2
    
    # Estimate V_dot (derivative of Lyapunov function) using numerical differentiation
    V_dot_surge = np.gradient(V_surge, t_vec)
    V_dot_yaw = np.gradient(V_yaw, t_vec)
    
    # Plot Lyapunov function and its derivative
    ax_lyap[0].plot(t_vec, V_surge, 'b-', label='V (Surge)')
    ax_lyap[0].plot(t_vec, V_yaw, 'r-', label='V (Yaw)')
    ax_lyap[0].set_title('Lyapunov Function')
    ax_lyap[0].set_xlabel('Time [s]')
    ax_lyap[0].set_ylabel('V')
    ax_lyap[0].grid(True)
    ax_lyap[0].legend()
    
    ax_lyap[1].plot(t_vec, V_dot_surge, 'b-', label='dV/dt (Surge)')
    ax_lyap[1].plot(t_vec, V_dot_yaw, 'r-', label='dV/dt (Yaw)')
    ax_lyap[1].plot(t_vec, np.zeros_like(t_vec), 'k--')  # Zero line
    ax_lyap[1].set_title('Derivative of Lyapunov Function')
    ax_lyap[1].set_xlabel('Time [s]')
    ax_lyap[1].set_ylabel('dV/dt')
    ax_lyap[1].grid(True)
    ax_lyap[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 4. Control Effectiveness Analysis
    fig_control, ax_control = plt.subplots(2, 1, figsize=(14, 10))
    
    # Calculate control energy
    control_energy_u = np.cumsum(control_u**2) * (t_vec[1] - t_vec[0])
    control_energy_r = np.cumsum(control_r**2) * (t_vec[1] - t_vec[0])
    
    ax_control[0].plot(t_vec, control_energy_u, 'g-', label='Surge Control Energy')
    ax_control[0].plot(t_vec, control_energy_r, 'm-', label='Yaw Control Energy')
    ax_control[0].set_title('Control Energy')
    ax_control[0].set_xlabel('Time [s]')
    ax_control[0].set_ylabel('Cumulative Energy')
    ax_control[0].grid(True)
    ax_control[0].legend()
    
    # Calculate control switching frequency (by estimating sign changes)
    switch_u = np.abs(np.diff(np.sign(control_u)))
    switch_r = np.abs(np.diff(np.sign(control_r)))
    switch_freq_u = switch_u.cumsum() / t_vec[1:-1]
    switch_freq_r = switch_r.cumsum() / t_vec[1:-1]
    
    ax_control[1].plot(t_vec[1:-1], switch_freq_u, 'g-', label='Surge Control Switching')
    ax_control[1].plot(t_vec[1:-1], switch_freq_r, 'm-', label='Yaw Control Switching')
    ax_control[1].set_title('Control Switching Frequency')
    ax_control[1].set_xlabel('Time [s]')
    ax_control[1].set_ylabel('Switches/s')
    ax_control[1].grid(True)
    ax_control[1].legend()
    
    plt.tight_layout()
    plt.show()
    
# 6. Integrasi Dinamika Sistem
def system_dynamics(eta, nu, forces, moments, control, hull, simulation):
    """
    Menghitung perubahan keadaan (nudot, etadot) berdasarkan dinamika sistem.
    Parameters:
        eta (np.array): [x, y, psi]
        nu (np.array): [u, v, r]
        forces (dict): Koefisien gaya
        moments (dict): Koefisien momen
        control (dict): Parameter kontrol
        hull (dict): Parameter hull
        simulation (dict): Parameter simulasi
    Returns:
        nudot (np.array): Perubahan kecepatan
        etadot (np.array): Perubahan posisi dan orientasi
    """
    # Compute hydrodynamic forces and moments based on current velocities
    ur = nu[0]
    vr = nu[1]
    r = nu[2]
    # Gaya X (Surge)
    F_X = (forces['X_uu'] * ur * abs(ur) +
           forces['X_wq'] * r * abs(r) +
           forces['X_qq'] * r**2 * ur)
    # Gaya Y (Sway)
    F_Y = (forces['Y_vv'] * vr * abs(vr) +
           forces['Y_rr'] * r**2 * vr +
           forces['Y_uv'] * ur * vr +
           forces['Y_v'] * vr +
           forces['Y_r'] * r +
           forces['Y_ur'] * ur * r +
           forces['Y_wp'] * r +
           forces['Y_pq'] * r**2 * ur +
           forces['Y_udr'] * 0.0)  # Asumsi Fin Lift Force 2D = 0
    # Momen Z (Yaw)
    M_Z = (moments['N_rr'] * r**2 * vr +
           moments['N_uw'] * ur * vr +
           moments['N_v'] * vr +
           moments['N_r'] * r +
           moments['N_ur'] * ur * r +
           moments['N_wp'] * r +
           moments['N_pq'] * r**2 * ur +
           moments['N_uadr'] * 0.0)  # Asumsi Fin Lift Moment 2D = 0
    # Total Gaya dan Momen
    F = np.array([F_X, F_Y, M_Z])
    # Get control input
    if 'u' in control:
        tau = control['u']  # [Surge, Yaw] control input
        # Expand to 3D if needed
        if len(tau) == 2:
            tau = np.array([tau[0], 0.0, tau[1]])  # [surge, sway, yaw]
    else:
        tau = np.array([0.0, 0.0, 0.0])
    # Improved mass matrix
    m = 15.0  # kg, increased mass for better stability
    I_z = 2.0  # kg*m^2, increased moment of inertia
    M_matrix = np.diag([m, m, I_z])
    
    # Add stronger nonlinear damping for better stability
    surge_damping = 1.0 + 0.8 * abs(nu[0])  # Nonlinear damping increases with speed
    sway_damping = 1.5 + 1.0 * abs(nu[1])
    yaw_damping = 2.0 + 1.2 * abs(nu[2])
    
    F[0] -= surge_damping * nu[0]  # Surge damping
    F[1] -= sway_damping * nu[1]   # Sway damping  
    F[2] -= yaw_damping * nu[2]    # Yaw damping
    
    # Compute nudot = M^{-1} * (tau - F)
    nudot = np.linalg.inv(M_matrix) @ (tau - F)
    
    # Limit acceleration for stability
    max_accel = 1.0  # m/s^2 - reduced for better stability
    max_yaw_accel = 0.5  # rad/s^2 - limit yaw acceleration specifically
    
    nudot[0:2] = np.clip(nudot[0:2], -max_accel, max_accel)  # Limit surge/sway accel
    nudot[2] = np.clip(nudot[2], -max_yaw_accel, max_yaw_accel)  # Limit yaw accel
    # Compute etadot
    R = np.array([
        [np.cos(eta[2]), -np.sin(eta[2])],
        [np.sin(eta[2]),  np.cos(eta[2])]
    ])
    etadot = np.array([
        R[0,0] * nu[0] + R[0,1] * nu[1],
        R[1,0] * nu[0] + R[1,1] * nu[1],
        nu[2]
    ])
    return nudot, etadot

# 7. Simulasi dan Visualisasi
def auv_sliding_mode_simulation():
    # Inisialisasi Parameter
    hull, forces, moments, control, simulation = initialize_parameters()
    # Inisialisasi Kondisi Awal
    eta = np.array([0.0, 0.0, 0.0])    # [x, y, psi] - start at origin
    nu = np.array([0.1, 0.0, 0.0])     # [u, v, r] - small initial velocity
    wpIdx = 0
    # Inisialisasi Log Data
    N_sim = int(np.ceil(simulation['Tend'] / simulation['dt'])) + 1
    traj = np.full((N_sim, 5), np.nan)   # x, y, u, v, r
    ct_err = np.full(N_sim, np.nan)
    hdg_err = np.full(N_sim, np.nan)
    t_vec = np.full(N_sim, np.nan)
    control_u = np.full(N_sim, np.nan)
    control_r = np.full(N_sim, np.nan)
    # Sliding Manifold Logging
    s_surge = np.full(N_sim, np.nan)
    s_yaw = np.full(N_sim, np.nan)
    # === Plot Setup ===
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    # Subplot AUV XY Plane
    ax1 = axs[0, 0]
    ax1.set_title('AUV XY Plane')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.plot(simulation['waypoints'][:,0], simulation['waypoints'][:,1], 'k--', linewidth=1.5, label='Waypoints')
    ax1.plot(simulation['waypoints'][:,0], simulation['waypoints'][:,1], 'ko', markerfacecolor='k')
    ax1.set_xlim(-5, 40)
    ax1.set_ylim(-5, 30)
    ax1.grid(True)
    auv_shape, = ax1.plot([], [], 'b-')  # AUV shape
    traj_line, = ax1.plot([], [], 'r-', label='Trajectory')
    # Subplot Cross-Track Error
    ax2 = axs[0, 1]
    ax2.set_title('Cross-Track Error')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('e_ct [m]')
    ax2.set_ylim(-5, 5)
    ax2.grid(True)
    err_ct_line, = ax2.plot([], [], 'b-')
    # Subplot Heading Error
    ax3 = axs[1, 0]
    ax3.set_title('Heading Error')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('e_psi [deg]')
    ax3.set_ylim(-180, 180)
    ax3.grid(True)
    err_hdg_line, = ax3.plot([], [], 'r-')
    # Subplot Control Inputs
    ax4 = axs[1, 1]
    ax4.set_title('Control Inputs')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Control [N/Rad]')
    ax4.set_ylim(-100, 100)
    ax4.grid(True)
    tau_u_line, = ax4.plot([], [], 'g-', label='Surge (tau_u)')
    tau_r_line, = ax4.plot([], [], 'm-', label='Yaw (tau_r)')
    ax4.legend()
    plt.tight_layout()
    # === Simulasi Loop ===
    # Initialize disturbance tracking
    disturbance_force = np.zeros((N_sim, 2))  # Track disturbance for stability analysis
    
    for k in range(N_sim):
        t = k * simulation['dt']
        t_vec[k] = t
        # --- LOS Guidance ---
        psi_d, e_ct, wpIdx = los_guidance(eta, simulation['waypoints'], wpIdx, simulation)
        ct_err[k] = e_ct
        # --- Control Law ---
        e_psi = wrap_to_pi(eta[2] - psi_d)
        hdg_err[k] = np.degrees(e_psi)
        
        # Calculate error derivatives
        u_err_derivative = 0.0  # For initial simplicity
        if k > 0:
            e_psi_prev = np.radians(hdg_err[k-1])
            psi_err_derivative = (e_psi - e_psi_prev) / simulation['dt']
            psi_err_derivative = np.clip(psi_err_derivative, -1.0, 1.0)  # Limit derivative for stability
        else:
            psi_err_derivative = 0.0
        
        # Sliding Mode Control untuk Surge - batas kecepatan
        tau_u, s_u = sliding_mode_control(simulation['u_des'], nu[0], u_err_derivative, control, 'surge')
        
        # Sliding Mode Control untuk Yaw - menggunakan 0.0 sebagai referensi untuk error
        tau_r, s_r = sliding_mode_control(0.0, e_psi, psi_err_derivative, control, 'yaw')
        # Simulasi gangguan lingkungan untuk testing robustness SMC
        # Menggunakan gangguan yang lebih halus dan realistis
        current_disturbance = np.array([0.0, 0.0])
        
        # Gangguan arus laut - realistis dengan amplitudo yang lebih kecil
        if 20 <= t < 30:  # Gangguan 1: Ocean current
            # Sinusoidal disturbance untuk simulasi gelombang
            wave_freq = 0.5  # Hz
            dist_surge = 0.1 * np.sin(2 * np.pi * wave_freq * t)
            dist_yaw = 0.02 * np.sin(2 * np.pi * wave_freq * t + np.pi/4)
            current_disturbance = np.array([dist_surge, dist_yaw])
        elif 60 <= t < 70:  # Gangguan 2: Thruster failure simulation - lebih halus
            # Gangguan thruster yang bertahap, bukan tiba-tiba
            ramp_factor = min(1.0, (t - 60) / 2.0)  # Ramp up over 2 seconds
            dist_surge = -0.1 * ramp_factor
            dist_yaw = -0.02 * ramp_factor
            current_disturbance = np.array([dist_surge, dist_yaw])
        
        # Store disturbance for stability analysis
        disturbance_force[k] = current_disturbance
            
        # Tambahkan gangguan ke kontrol dengan faktor amplitudo yang lebih kecil
        tau_u += current_disturbance[0]
        tau_r += current_disturbance[1]
        # Assign kontrol input
        u_control = np.array([tau_u, tau_r])
        # Store control input and sliding manifold
        control['u'] = u_control
        control_u[k] = tau_u
        control_r[k] = tau_r
        s_surge[k] = s_u
        s_yaw[k] = s_r
        # --- Sistem Dinamika ---
        nudot, etadot = system_dynamics(eta, nu, forces, moments, control, hull, simulation)
        # Update state with velocity limits - using different limits for each component
        nu += nudot * simulation['dt']
        # Set different limits for surge, sway, and yaw velocities
        nu[0] = np.clip(nu[0], -0.8, 0.8)  # Limit surge velocity
        nu[1] = np.clip(nu[1], -0.5, 0.5)  # Limit sway velocity
        nu[2] = np.clip(nu[2], -0.3, 0.3)  # Limit yaw rate for stability
        eta += etadot * simulation['dt']
        # Log Trajectory
        traj[k, 0:2] = eta[0:2]
        traj[k, 2:5] = nu
        # --- Visualisasi ---
        if k % 10 == 0:
            # Update AUV Shape
            L = 1.2  # Length
            W = 0.5  # Width
            shape = np.array([
                [L, -L/2, -L/2],
                [0, -W/2, W/2]
            ])
            R_mat = np.array([
                [np.cos(eta[2]), -np.sin(eta[2])],
                [np.sin(eta[2]),  np.cos(eta[2])]
            ])
            shape_rot = R_mat @ shape
            shape_rot += eta[0:2].reshape(2,1)
            auv_shape.set_data(shape_rot[0,:], shape_rot[1,:])
            # Update Trajectory
            traj_line.set_data(traj[:k+1,0], traj[:k+1,1])
            # Update Errors
            err_ct_line.set_data(t_vec[:k+1], ct_err[:k+1])
            err_hdg_line.set_data(t_vec[:k+1], hdg_err[:k+1])
            # Update Control Inputs
            tau_u_line.set_data(t_vec[:k+1], control_u[:k+1])
            tau_r_line.set_data(t_vec[:k+1], control_r[:k+1])
            # Update plot limits
            if k > 0:
                ax2.set_xlim(0, t + simulation['dt'])
                ax3.set_xlim(0, t + simulation['dt'])
                ax4.set_xlim(0, t + simulation['dt'])
            plt.pause(0.001)  # Pause kecil untuk update plot
        # Check for simulation termination
        if wpIdx >= len(simulation['waypoints'])-1 and e_ct == 0.0:
            print(f"✅ Simulasi selesai pada waktu t = {t:.2f} detik")
            break
    # === Plotting Akhir ===
    plt.show()
    # === Plot Error dan Kontrol Inputs secara Terpisah ===
    fig2, axs2 = plt.subplots(2, 1, figsize=(14, 10))
    axs2[0].plot(t_vec, ct_err, 'b-', label='Cross-Track Error')
    axs2[0].set_title('Cross-Track Error over Time')
    axs2[0].set_xlabel('Time [s]')
    axs2[0].set_ylabel('e_ct [m]')
    axs2[0].grid(True)
    axs2[0].legend()
    axs2[1].plot(t_vec, hdg_err, 'r-', label='Heading Error')
    axs2[1].set_title('Heading Error over Time')
    axs2[1].set_xlabel('Time [s]')
    axs2[1].set_ylabel('e_psi [deg]')
    axs2[1].grid(True)
    axs2[1].legend()
    plt.tight_layout()
    plt.show()
    # === Plot Control Inputs ===
    fig3, axs3 = plt.subplots(2, 1, figsize=(14, 10))
    axs3[0].plot(t_vec, control_u, 'g-', label='Surge Control (tau_u)')
    axs3[0].set_title('Surge Control Input over Time')
    axs3[0].set_xlabel('Time [s]')
    axs3[0].set_ylabel('tau_u [N]')
    axs3[0].grid(True)
    axs3[0].legend()
    axs3[1].plot(t_vec, control_r, 'm-', label='Yaw Control (tau_r)')
    axs3[1].set_title('Yaw Control Input over Time')
    axs3[1].set_xlabel('Time [s]')
    axs3[1].set_ylabel('tau_r [Nm]')
    axs3[1].grid(True)
    axs3[1].legend()
    plt.tight_layout()
    plt.show()
    # === Plot Sliding Manifold ===
    fig4, axs4 = plt.subplots(2, 1, figsize=(14, 10))
    axs4[0].plot(t_vec, s_surge, 'b-', label='Sliding Surface Surge (s_u)')
    axs4[0].set_title('Sliding Surface Surge over Time')
    axs4[0].set_xlabel('Time [s]')
    axs4[0].set_ylabel('s_u')
    axs4[0].grid(True)
    axs4[0].legend()
    axs4[1].plot(t_vec, s_yaw, 'r-', label='Sliding Surface Yaw (s_r)')
    axs4[1].set_title('Sliding Surface Yaw over Time')
    axs4[1].set_xlabel('Time [s]')
    axs4[1].set_ylabel('s_r')
    axs4[1].grid(True)
    axs4[1].legend()
    plt.tight_layout()
    plt.show()
    # === Performance Analysis ===
    print("\n" + "="*60)
    print("SLIDING MODE CONTROL PERFORMANCE ANALYSIS")
    print("="*60)
    # Calculate MSE
    mse_ct = np.nanmean(ct_err**2)
    mse_hdg = np.nanmean(hdg_err**2)
    # Control effort analysis
    avg_control_u = np.nanmean(np.abs(control_u))
    avg_control_r = np.nanmean(np.abs(control_r))
    max_control_u = np.nanmax(np.abs(control_u))
    max_control_r = np.nanmax(np.abs(control_r))
    print(f"Cross-Track Error MSE: {mse_ct:.4f}")
    print(f"Heading Error MSE: {mse_hdg:.4f}")
    print(f"\nControl Effort Analysis:")
    print(f"Average Surge Control: {avg_control_u:.2f} N")
    print(f"Average Yaw Control: {avg_control_r:.2f} Nm")
    print(f"Maximum Surge Control: {max_control_u:.2f} N")
    print(f"Maximum Yaw Control: {max_control_r:.2f} Nm")
    
    # === Stability Analysis ===
    print("\n" + "="*60)
    print("LYAPUNOV STABILITY ANALYSIS")
    print("="*60)
    
    # Extract valid data points (non-NaN values)
    valid_idx = ~np.isnan(s_surge) & ~np.isnan(s_yaw) & ~np.isnan(t_vec)
    
    if np.any(valid_idx):
        # Perform Lyapunov stability analysis
        # First, create a state representation for analysis
        state_history = np.column_stack((traj[valid_idx, 0:2], np.radians(hdg_err[valid_idx])))
        
        # Calculate convergence rates
        converge_ct = np.nanmean(np.abs(np.diff(ct_err[valid_idx])))
        converge_hdg = np.nanmean(np.abs(np.diff(hdg_err[valid_idx])))
        
        print(f"Cross-Track Error Convergence Rate: {converge_ct:.6f} m/step")
        print(f"Heading Error Convergence Rate: {converge_hdg:.6f} deg/step")
        
        # Calculate stability metrics based on sliding surfaces
        s_surge_stability = np.nanmean(np.sign(s_surge[valid_idx]) * 
                                      np.sign(np.gradient(s_surge[valid_idx])))
        s_yaw_stability = np.nanmean(np.sign(s_yaw[valid_idx]) * 
                                    np.sign(np.gradient(s_yaw[valid_idx])))
        
        print("\nSliding Surface Stability Analysis:")
        print(f"Surge Sliding Surface Stability Metric: {s_surge_stability:.4f}")
        print(f"Yaw Sliding Surface Stability Metric: {s_yaw_stability:.4f}")
        print("Note: Negative values indicate convergence towards the sliding surface")
        
        # Theoretical stability guarantee based on control parameters
        lambda_min_KS = min(control['KS'][0, 0], control['KS'][1, 1])
        estimated_disturbance_bound = max(np.nanmax(np.abs(disturbance_force[0])), 
                                         np.nanmax(np.abs(disturbance_force[1])))
        
        print("\nTheoretical Stability Guarantee:")
        print(f"Min eigenvalue of KS: {lambda_min_KS:.2f}")
        print(f"Estimated disturbance bound: {estimated_disturbance_bound:.4f}")
        
        if lambda_min_KS > estimated_disturbance_bound:
            print("✅ System satisfies Lyapunov stability condition (λ_min(KS) > disturbance bound)")
        else:
            print("⚠️ Stability condition might be violated (λ_min(KS) ≤ disturbance bound)")
            print("   Consider increasing KS gains or improving disturbance rejection")
            
        # Visualize phase portrait and Lyapunov stability
        visualize_sliding_mode_stability(t_vec[valid_idx], s_surge[valid_idx], 
                                        s_yaw[valid_idx], state_history, 
                                        traj[valid_idx, 2:5], control_u[valid_idx], 
                                        control_r[valid_idx])

# Jalankan simulasi
if __name__ == "__main__":
    auv_sliding_mode_simulation()
