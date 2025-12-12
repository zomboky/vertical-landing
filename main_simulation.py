import pybullet as p
import pybullet_data
import numpy as np
import time

# --- IMPORT DU CONTROLEUR ---
from flight_controller import FlightComputer, RocketParams

# 1. INITIALISATION
config = RocketParams()
computer = FlightComputer()  # Notre cerveau importé

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -config.g)
p.loadURDF("plane.urdf")

# Création Fusée
col = p.createCollisionShape(p.GEOM_CYLINDER, radius=config.W / 2, height=config.H)
vis = p.createVisualShape(p.GEOM_CYLINDER, radius=config.W / 2, length=config.H, rgbaColor=[1, 1, 1, 1])

# On la place haut et loin pour tester le PID
start_pos = [30, -20, 600]
rocket_id = p.createMultiBody(baseMass=computer.current_mass,
                              baseCollisionShapeIndex=col,
                              baseVisualShapeIndex=vis,
                              basePosition=start_pos)

# Friction pour l'atterrissage
p.changeDynamics(rocket_id, -1, lateralFriction=1.0, spinningFriction=0.1, rollingFriction=0.1)

# Paramètre Nozzle (bas de la fusée)
nozzle_pos_local = [0, 0, -config.H / 2]
dt = 1 / 240

# 2. BOUCLE
print("Lancement simulation...")
for step in range(100000):

    # --- A. CAPTEURS (PYBULLET -> DATA) ---
    pos, orn = p.getBasePositionAndOrientation(rocket_id)
    lin_vel, ang_vel = p.getBaseVelocity(rocket_id)
    euler = p.getEulerFromQuaternion(orn)

    # --- B. CERVEAU (DATA -> COMMANDES) ---
    # On demande au contrôleur quoi faire
    thrust, torques = computer.update(pos, lin_vel, euler, ang_vel, dt)

    # --- C. ACTIONNEURS (COMMANDES -> PYBULLET) ---

    # 1. Mise à jour masse physique
    p.changeDynamics(rocket_id, -1, mass=computer.current_mass)

    # 2. Appliquer la Poussée
    if thrust > 0:
        # Orientation actuelle pour diriger la poussée
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        thrust_vec = R @ [0, 0, thrust]  # Force locale Z vers Monde
        p.applyExternalForce(rocket_id, -1, thrust_vec, nozzle_pos_local, p.LINK_FRAME)

    # 3. Appliquer les Couples (Rotation)
    p.applyExternalTorque(rocket_id, -1, torques, p.WORLD_FRAME)

    # --- D. VISUEL ---
    p.resetDebugVisualizerCamera(
        cameraDistance=25 + pos[2] * 0.2,
        cameraYaw=45, cameraPitch=-15,
        cameraTargetPosition=pos
    )

    p.stepSimulation()
    time.sleep(dt)

p.disconnect()