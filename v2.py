import pybullet as p
import pybullet_data
import numpy as np
import time

# --------------------------------------------------------------------------- #
#                           PARAMÈTRES DE LA FUSÉE                            #
# --------------------------------------------------------------------------- #
class Rocket:
    def __init__(self):
        self.dt = 1/240
        self.g = 9.81
        self.H = 13.0
        self.W = 1.1
        self.mass = 1900
        self.I = (1/12) * self.mass * (3*(self.W/2)**2 + self.H**2)
        self.nb_moteurs = 9
        self.F_max = 4600 * self.nb_moteurs
        self.F_min = 1600 * self.nb_moteurs
        self.nozzle_pos = [0, 0, -self.H/2]

rocket = Rocket()

# --------------------------------------------------------------------------- #
#                           INITIALISATION PYBULLET                           #
# --------------------------------------------------------------------------- #
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -rocket.g)
plane_id = p.loadURDF("plane.urdf")

col = p.createCollisionShape(p.GEOM_CYLINDER, radius=rocket.W/2, height=rocket.H)
vis = p.createVisualShape(p.GEOM_CYLINDER, radius=rocket.W/2, length=rocket.H, rgbaColor=[1,1,1,1])
rocket_id = p.createMultiBody(baseMass=rocket.mass,
                              baseCollisionShapeIndex=col,
                              baseVisualShapeIndex=vis,
                              basePosition=[0, 0, 50],
                              baseInertialFramePosition=[0, 0, 0])

# --------------------------------------------------------------------------- #
#                           BOUCLE DE SIMULATION                               #
# --------------------------------------------------------------------------- #
for step in range(20000):
    # Récupérer position et orientation
    pos, orn = p.getBasePositionAndOrientation(rocket_id)
    R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    z = pos[2]

    # ----------------------- POUSSÉE POUR ATTERRISSAGE -------------------------
    # Calculer poussée pour contrer la gravité + ajustement selon altitude
    # Plus la fusée est proche du sol, plus la poussée augmente pour ralentir
    if z > 2:  # hauteur minimale d'arrêt
        thrust = rocket.mass * rocket.g * 0.90  # légèrement > poids
        # on peut réduire un peu la poussée quand z est très élevé
        thrust *= (1 + 0.5*(5 - min(z, 5))/5)  # ajustement en fonction de l'altitude
        gimbal = 0.0
    else:
        thrust = 0
        gimbal = 0

    # ------------------------------------------------------------------------------
    # Calcul vecteur de poussée
    gimbal_rot = p.getQuaternionFromEuler([gimbal, 0, 0])
    R_gimbal = np.array(p.getMatrixFromQuaternion(gimbal_rot)).reshape(3,3)
    thrust_dir_local = R_gimbal @ np.array([0, 0, 1])
    thrust_world = R @ thrust_dir_local
    thrust_force = thrust * thrust_world

    # Appliquer force moteur
    p.applyExternalForce(rocket_id, -1, forceObj=thrust_force, posObj=rocket.nozzle_pos, flags=p.LINK_FRAME)

    # Appliquer couple moteur
    torque_local = np.array([gimbal * thrust * rocket.H / 2, 0, 0])
    torque_world = R @ torque_local
    p.applyExternalTorque(rocket_id, -1, torqueObj=torque_world, flags=p.WORLD_FRAME)

    # --------------------------------------------------------------------------- #
    #                           CAMÉRA JETPACK                                   #
    # --------------------------------------------------------------------------- #
    cam_local = np.array([0, -20, 5])
    cam_world = pos + R @ cam_local
    p.resetDebugVisualizerCamera(
        cameraDistance=np.linalg.norm(cam_world - pos),
        cameraYaw=np.degrees(np.arctan2(cam_world[1]-pos[1], cam_world[0]-pos[0])),
        cameraPitch=-np.degrees(np.arctan2(cam_world[2]-pos[2], np.linalg.norm(cam_world[:2]-pos[:2]))),
        cameraTargetPosition=pos
    )

    # --------------------------------------------------------------------------- #
    # Avancer la simulation
    p.stepSimulation()
    time.sleep(rocket.dt)

p.disconnect()






