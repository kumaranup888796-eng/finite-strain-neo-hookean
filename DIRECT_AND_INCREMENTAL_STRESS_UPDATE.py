" ASSIGNMENT-6 CALCAULTION OF STRESS TENSORS FOR THE GIVEN MAP"
import numpy as np
import matplotlib.pyplot as plt

# Material Parameters
Em=210e3;nu=0.2
mu=Em/(2*(1 + nu));lam = Em*nu/((1 + nu)*(1-2*nu))

# We have '8' Noded Hexaderal Element in Natural Coordinate System!
nodes_zeta = np.array([
    [-1, -1, -1],   # Node 1
    [ 1, -1, -1],   # Node 2
    [ 1,  1, -1],   # Node 3
    [-1,  1, -1],   # Node 4
    [-1, -1,  1],   # Node 5
    [ 1, -1,  1],   # Node 6
    [ 1,  1,  1],   # Node 7
    [-1,  1,  1]    # Node 8
], dtype=float)    # Coordinate Values of each Nodal point!!

# Reference Material Nodal Coordinate values (3X8)
X_nodal=nodes_zeta.T 

# Defining Shape functions
def shape_function(zeta1, zeta2, zeta3):
    N = np.zeros((8, 1))   
    N[0, 0] = (1/8) * (1 - zeta1) * (1 - zeta2) * (1 - zeta3)   # Node 1
    N[1, 0] = (1/8) * (1 + zeta1) * (1 - zeta2) * (1 - zeta3)   # Node 2
    N[2, 0] = (1/8) * (1 + zeta1) * (1 + zeta2) * (1 - zeta3)   # Node 3
    N[3, 0] = (1/8) * (1 - zeta1) * (1 + zeta2) * (1 - zeta3)   # Node 4
    N[4, 0] = (1/8) * (1 - zeta1) * (1 - zeta2) * (1 + zeta3)   # Node 5
    N[5, 0] = (1/8) * (1 + zeta1) * (1 - zeta2) * (1 + zeta3)   # Node 6
    N[6, 0] = (1/8) * (1 + zeta1) * (1 + zeta2) * (1 + zeta3)   # Node 7
    N[7, 0] = (1/8) * (1 - zeta1) * (1 + zeta2) * (1 + zeta3)   # Node 8
    return N #Matrix Output : 8 x 1

# Defining Shape function derivatives!
def shape_func_derivatives(zeta1, zeta2, zeta3):
    delN_delzeta = np.zeros((8, 3))   # 8 nodes and 3 natural coordinate derivatives
    delN_delzeta[0, 0] = -(1/8) * (1 - zeta2) * (1 - zeta3)   # delN1/dzeta1
    delN_delzeta[0, 1] = -(1/8) * (1 - zeta1) * (1 - zeta3)   # delN1/dzeta2
    delN_delzeta[0, 2] = -(1/8) * (1 - zeta1) * (1 - zeta2)   # delN1/dzeta3
    delN_delzeta[1, 0] =  (1/8) * (1 - zeta2) * (1 - zeta3)   # delN2/dzeta1
    delN_delzeta[1, 1] = -(1/8) * (1 + zeta1) * (1 - zeta3)   # delN2/dzeta2
    delN_delzeta[1, 2] = -(1/8) * (1 + zeta1) * (1 - zeta2)   # delN2/dzeta3
    delN_delzeta[2, 0] =  (1/8) * (1 + zeta2) * (1 - zeta3)   # delN3/dzeta1
    delN_delzeta[2, 1] =  (1/8) * (1 + zeta1) * (1 - zeta3)   # delN3/dzeta2
    delN_delzeta[2, 2] = -(1/8) * (1 + zeta1) * (1 + zeta2)   # delN3/dzeta3
    delN_delzeta[3, 0] = -(1/8) * (1 + zeta2) * (1 - zeta3)   # delN4/dzeta1
    delN_delzeta[3, 1] =  (1/8) * (1 - zeta1) * (1 - zeta3)   # delN4/dzeta2
    delN_delzeta[3, 2] = -(1/8) * (1 - zeta1) * (1 + zeta2)   # delN4/dzeta3
    delN_delzeta[4, 0] = -(1/8) * (1 - zeta2) * (1 + zeta3)   # delN5/dzeta1
    delN_delzeta[4, 1] = -(1/8) * (1 - zeta1) * (1 + zeta3)   # delN5/dzeta2
    delN_delzeta[4, 2] =  (1/8) * (1 - zeta1) * (1 - zeta2)   # delN5/dzeta3
    delN_delzeta[5, 0] =  (1/8) * (1 - zeta2) * (1 + zeta3)   # delN6/dzeta1
    delN_delzeta[5, 1] = -(1/8) * (1 + zeta1) * (1 + zeta3)   # delN6/dzeta2
    delN_delzeta[5, 2] =  (1/8) * (1 + zeta1) * (1 - zeta2)   # delN6/dzeta3
    delN_delzeta[6, 0] =  (1/8) * (1 + zeta2) * (1 + zeta3)   # delN7/dzeta1
    delN_delzeta[6, 1] =  (1/8) * (1 + zeta1) * (1 + zeta3)   # delN7/dzeta2
    delN_delzeta[6, 2] =  (1/8) * (1 + zeta1) * (1 + zeta2)   # delN7/dzeta3
    delN_delzeta[7, 0] = -(1/8) * (1 + zeta2) * (1 + zeta3)   # delN8/dzeta1
    delN_delzeta[7, 1] =  (1/8) * (1 - zeta1) * (1 + zeta3)   # delN8/dzeta2
    delN_delzeta[7, 2] =  (1/8) * (1 - zeta1) * (1 + zeta2)   # delN8/dzeta3
    return delN_delzeta #Matrix Output : (8 x 3)

# Now as per Assignment evaluation has to be done at Nodal point(1,1,1)
N=shape_function(1,1,1) ;  delN_delzeta=shape_func_derivatives(1,1,1)

# Reference Material Nodal Coordinate values using Shape functions (3X1)
X_zeta_t = X_nodal@N   

# Now Lets calculate Jacobian Matrix using del(X(zeta,t)/delZeta)
J=X_nodal@delN_delzeta ; J_inv=np.linalg.inv(J)

# Lets Define all the Transformation Maps given in the question
# Map 1: Pure Translation
def Pure_Translation(X_nodal, t):
    X1,X2,X3=X_nodal
    x_nodal=np.zeros_like(X_nodal)
    x_nodal[0,:]=t+X1
    x_nodal[1,:]=t+X2
    x_nodal[2,:]=t+X3
    return x_nodal
"=============================================================================================================================================="

# Map 2: Rotation + Translation
def Rotation_translation(X_nodal, t):
    X1, X2, X3 = X_nodal
    x_nodal = np.zeros_like(X_nodal)
    theta = (np.pi / 6) * t
    x_nodal[0, :] = np.cos(theta)*X1-np.sin(theta)*X2 + t
    x_nodal[1, :] = np.sin(theta) * X1 + np.cos(theta)*X2 + t
    x_nodal[2, :] = X3
    return x_nodal

"=============================================================================================================================================="

# Map 3: Pure Shear
def Pure_shear(X_nodal, t):
    X1, X2, X3 = X_nodal
    x_nodal = np.zeros_like(X_nodal)
    x_nodal[0, :] = X1+t* X2
    x_nodal[1, :] = X2
    x_nodal[2, :] = X3
    return x_nodal

"=============================================================================================================================================="

# Map 4: Generic Deformation Map
def Generic_map(X_nodal, t):
    X1, X2, X3 = X_nodal
    x_nodal = np.zeros_like(X_nodal)
    et = np.exp(t)
    emt = np.exp(-t)
    x_nodal[0, :] = et * X1 + (et - 1) * X3
    x_nodal[1, :] = X2 + (et - emt) * X3
    x_nodal[2, :] = X3
    return x_nodal

"=============================================================================================================================================="

# Map 5: Inverted Map
def Inverted_map(X_nodal, t):
    X1, X2, X3 = X_nodal
    x_nodal = np.zeros_like(X_nodal)
    et = np.exp(t)
    emt = np.exp(-t)
    x_nodal[0, :] = et * X1 + (et - 1) * X3
    x_nodal[1, :] = X2 + (et - emt) * X3
    x_nodal[2, :] = X3
    return x_nodal

"=============================================================================================================================================="

# Choosing the Deformation Mapping part!!
print("Choose deformation map:")
print("1 = Pure Translation")
print("2 = Rotation + Translation")
print("3 = Pure Shear")
print("4 = Generic Map")
print("5 = Inverted Map")

choice = int(input("Enter your choice: "))
maps = {
    1: [Pure_Translation, "Pure Translation"],
    2: [Rotation_translation, "Rotation + Translation"],
    3: [Pure_shear, "Pure Shear"],
    4: [Generic_map, "Generic Map"],
    5: [Inverted_map, "Inverted Map"]
}
selected_map, map_name = maps[choice]
print("Selected Map:", map_name)


"=============================================================================================================================================="
# Computation of all Required Tensors using Direct Approach!
I = np.eye(3)
# To clear Residual Errors issues!
def Residual(value):
    if abs(value)<1e-10:
        return 0.0
    return value

def Tensors_Calculation(t):
    # 1. Current nodal coordinates from selected map!
    x_nodal = selected_map(X_nodal, t)

    # 2. Nodal displacement
    u_nodal = x_nodal-X_nodal

    # 3. Displacement gradient
    delu_delzeta = u_nodal @ delN_delzeta
    delu_delX = delu_delzeta @ J_inv

    # 4. Deformation gradient
    F = I + delu_delX

    # 5. Strain tensors
    C = F.T @ F # Right Cauchy-Green Tensor
    E = 0.5 * (C - I) # Lagrangian Strain Tensor
    b = F @ F.T # Left Cauchy-Green Tensor
    e = 0.5 * (I-np.linalg.inv(b)) # Euler Strain Tensor

    # 6. Second Piola Stress Tensor for Neo-Hookean Material
    JF = np.linalg.det(F)
    C_inv = np.linalg.inv(C)
    S = mu*(I-C_inv)+lam*np.log(JF)*C_inv

    # 7. Cauchy stress Tensor 
    sigma = (1 / JF) * F @ S @ F.T

    # 8. Equivalent stress values by substracting deviatoric part
    S_dev = S-(1/3)*np.trace(S)*I ;  sigma_dev = sigma - (1/3) * np.trace(sigma)*I
    Seq = np.sqrt((3/2)*np.sum(S_dev*S_dev))
    sigma_eq = np.sqrt((3/2) * np.sum(sigma_dev * sigma_dev))

    # 9. Equivalent strain norms
    E_norm = (1/(np.sqrt(2) * (1 + nu)))*np.sqrt(
        (E[0,0] - E[1,1])**2 +
        (E[1,1] - E[2,2])**2 +
        (E[2,2] - E[0,0])**2 +
        3*(E[0,1]**2 + E[1,2]**2 + E[2,0]**2))

    e_norm = (1/(np.sqrt(2)*(1 + nu)))*np.sqrt(
        (e[0,0] - e[1,1])**2 +
        (e[1,1] - e[2,2])**2 +
        (e[2,2] - e[0,0])**2 +
        3*(e[0,1]**2 + e[1,2]**2 + e[2,0]**2))

    # 10. Remove tiny numerical errors for clean output!
    Seq = Residual(Seq)
    sigma_eq = Residual(sigma_eq)
    E_norm = Residual(E_norm)
    e_norm = Residual(e_norm)

    return F, C, E, b, e, S, sigma, Seq, sigma_eq, E_norm, e_norm


# Printing Tensor Results
def print_results(t, F, C, E, b, e, S, sigma, Seq, sigma_eq, E_norm, e_norm):

    print("\n====================================================")
    print("Time t =", t)
    print("====================================================")

    print("\nF =\n", F)
    print("\nC =\n", C)
    print("\nE =\n", E)
    print("\nb =\n", b)
    print("\ne =\n", e)
    print("\nS =\n", S)
    print("\nsigma =\n", sigma)

    print("\nSeq =", Seq)
    print("sigma_eq =", sigma_eq)
    print("||E|| =", E_norm)
    print("||e|| =", e_norm)



" =============================================================================================================================================="
# Incremental Approach Calcualtions!
def Material_Tangent(C_old, J_old): # Material Tangent Matrix needs to be updated at every time increament!
    C_inv = np.linalg.inv(C_old)
    Cmat = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    term1 = C_inv[i, k] * C_inv[j, l]
                    term2 = C_inv[i, l] * C_inv[j, k]
                    term3 = C_inv[i, j] * C_inv[k, l]
                    Cmat[i, j, k, l] = (mu - lam * np.log(J_old))* (term1 + term2)+lam*term3
    return Cmat

def Stress_Increment(Cmat, Delta_E): # For Calculating Stress Increment using Material Tangent Matrix at every time increament!
    Delta_S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    Delta_S[i, j] = Delta_S[i, j] + Cmat[i, j, k, l] * Delta_E[k, l]
    return Delta_S

# Equivalent values for updated incremental tensors!
def Equivalent_Values(S, sigma, E, e):
    S_dev = S - (1/3)*np.trace(S)*I
    sigma_dev = sigma - (1/3)*np.trace(sigma)*I
    Seq = np.sqrt((3/2)*np.sum(S_dev*S_dev))
    sigma_eq = np.sqrt((3/2)*np.sum(sigma_dev*sigma_dev))
    E_norm = (1/(np.sqrt(2)*(1 + nu))) * np.sqrt(
        (E[0,0] - E[1,1])**2 +
        (E[1,1] - E[2,2])**2 +
        (E[2,2] - E[0,0])**2 +
        3*(E[0,1]**2 + E[1,2]**2 + E[2,0]**2)
    )
    e_norm = (1/(np.sqrt(2)*(1 + nu))) * np.sqrt(
        (e[0,0] - e[1,1])**2 +
        (e[1,1] - e[2,2])**2 +
        (e[2,2] - e[0,0])**2 +
        3*(e[0,1]**2 + e[1,2]**2 + e[2,0]**2))
    Seq = Residual(Seq)
    sigma_eq = Residual(sigma_eq)
    E_norm = Residual(E_norm)
    e_norm = Residual(e_norm)
    return Seq, sigma_eq, E_norm, e_norm



def Incremental_Calculation():
    Seq_list = []
    sigma_eq_list = []
    E_norm_list = []
    e_norm_list = []
    # Initial values at t = 0
    F_old, C_old, E_old, b_old, e_old, S_old, sigma_old, Seq, sigma_eq, E_norm, e_norm = Tensors_Calculation(0.0)
    Seq_list.append(Seq)
    sigma_eq_list.append(sigma_eq)
    E_norm_list.append(E_norm)
    e_norm_list.append(e_norm)

    for i in range(1, len(time_values)):
        t = time_values[i]
        # Calculate new F from direct kinematics
        F_new, C_new, E_new, b_new, e_new, S_new_direct, sigma_new_direct, Seq_direct, sigma_eq_direct, E_norm_direct, e_norm_direct = Tensors_Calculation(t)
        # Increment in deformation gradient
        Delta_F = F_new - F_old
        # Update deformation gradient
        F_updated = F_old + Delta_F
        # Increment in C
        Delta_C = Delta_F.T @ F_old + F_old.T @ Delta_F + Delta_F.T @ Delta_F
        # Update C and E
        C_updated = C_old + Delta_C
        Delta_E = 0.5 * Delta_C
        E_updated = E_old + Delta_E
        # Update Eulerian strain
        b_updated = F_updated @ F_updated.T
        e_updated = 0.5*(I-np.linalg.inv(b_updated))
        # Material tangent using old state
        J_old = np.linalg.det(F_old)
        Cmat_old = Material_Tangent(C_old, J_old)
        # Stress increment
        Delta_S = Stress_Increment(Cmat_old, Delta_E)
        # Update Second Piola stress
        S_updated = S_old + Delta_S
        # Update Cauchy stress
        J_updated = np.linalg.det(F_updated)
        sigma_updated = (1 / J_updated) * F_updated @ S_updated @ F_updated.T

        # Equivalent values
        Seq, sigma_eq, E_norm, e_norm = Equivalent_Values(S_updated,sigma_updated,E_updated,e_updated)       
        Seq_list.append(Seq)
        sigma_eq_list.append(sigma_eq)
        E_norm_list.append(E_norm)
        e_norm_list.append(e_norm)

        # Store current values for next increment
        F_old = F_updated
        C_old = C_updated
        E_old = E_updated
        S_old = S_updated

    return (np.array(Seq_list),
        np.array(sigma_eq_list),
        np.array(E_norm_list),
        np.array(e_norm_list))

" =============================================================================================================================================="

# PLOTTING THE RESULTS!
def make_overlay_plot(x_direct, y_direct, x_incr, y_incr, xlabel, ylabel, title):
    plt.figure(figsize=(9, 7), dpi=120)
    plt.plot(x_direct, y_direct, "o-", linewidth=3, markersize=9,
             markeredgecolor="black", label="DIRECT APPROACH")
    plt.plot(x_incr, y_incr, "s--", linewidth=3, markersize=9,
             markeredgecolor="black", label="INCREMENTAL APPROACH")
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title + "\n" + map_name, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(fontsize=16)
    plt.tight_layout()

    if np.all(x_direct == 0) and np.all(y_direct == 0) and np.all(x_incr == 0) and np.all(y_incr == 0):
        plt.xlim(-0.1, 0.1)
        plt.ylim(-0.1, 0.1)
    plt.show()

# Main Calculation
time_values = np.arange(0.0, 1, 0.01)
Seq_direct_list = []
sigma_eq_direct_list = []
E_norm_direct_list = []
e_norm_direct_list = []
for t in time_values:
    F, C, E, b, e, S, sigma, Seq, sigma_eq, E_norm, e_norm = Tensors_Calculation(t)
    print_results(t, F, C, E, b, e, S, sigma, Seq, sigma_eq, E_norm, e_norm)
    Seq_direct_list.append(Seq)
    sigma_eq_direct_list.append(sigma_eq)
    E_norm_direct_list.append(E_norm)
    e_norm_direct_list.append(e_norm)
Seq_direct_list = np.array(Seq_direct_list)
sigma_eq_direct_list = np.array(sigma_eq_direct_list)
E_norm_direct_list = np.array(E_norm_direct_list)
e_norm_direct_list = np.array(e_norm_direct_list)
Seq_incr_list, sigma_eq_incr_list, E_norm_incr_list, e_norm_incr_list = Incremental_Calculation()

# Plot Results
print("\n====================================================")
print("Map:", map_name)
print("Direct Approach vs Incremental Approach")
print("Evaluation point: zeta = (1,1,1)")
print("====================================================")

plot_data = [
    [E_norm_direct_list, Seq_direct_list, E_norm_incr_list, Seq_incr_list,
     r"$||\mathbf{E}||$", r"$S_{eq}$", r"$S_{eq}$ vs $||\mathbf{E}||$"],

    [e_norm_direct_list, sigma_eq_direct_list, e_norm_incr_list, sigma_eq_incr_list,
     r"$||\mathbf{e}||$", r"$\sigma_{eq}$", r"$\sigma_{eq}$ vs $||\mathbf{e}||$"]
]
for data in plot_data:
    make_overlay_plot(*data)

" =============================================================================================================================================="