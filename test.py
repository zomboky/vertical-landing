import pybullet as p
import pybullet_data
import time


p.connect(p.GUI)

# Paramètres de la Gravité 
p.setGravity(0, 0, -9.81)

# ---------------------------------------------------------------------------- #
#                                      Sol                                     #
# ---------------------------------------------------------------------------- #


p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")


# -------------------------------Fusée---------------------------------------- #

# Corps
collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=1.1, height=13)
visual_shape    = p.createVisualShape(p.GEOM_CYLINDER, radius=1.1, length=13, rgbaColor=[1, 1, 1, 1])


rocket_id = p.createMultiBody(
    baseMass=3400,
    baseCollisionShapeIndex=collision_shape,
    baseVisualShapeIndex=visual_shape,
    basePosition=[0, 0, 10],
)


# ---------------------------------------------------------------------------- #
#                              Simmulation + NOZZLE                            #
# ---------------------------------------------------------------------------- #

# Position du nozzle (collé au bas du cylindre)
NOZZLE_POS_LOCAL = [0, 0, -7]   # moitié de la longueur de 13 → -6.5, arrondi

# Force de poussée initiale (tu pourras la modifier dynamiquement)
nozzle_force = [0, 0, 30000]  # exemple : 50 kN vers +Z en frame local

for _ in range(1000):
    # --- Ajout du nozzle virtuel ---
    p.applyExternalForce(
        rocket_id,
        -1,                     # -1 = base
        nozzle_force,
        NOZZLE_POS_LOCAL,
        p.LINK_FRAME           # nozzle attaché à la fusée
    )
    # --------------------------------

    p.stepSimulation()
    time.sleep(1/100)

p.disconnect()
