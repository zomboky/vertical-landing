import pybullet as p
import pybullet_data
import numpy as np
import random
import time
from flight_controller import FlightComputer, RocketParams

# -----------------------------
# CONFIG GA / SIM
# -----------------------------
POPULATION_SIZE = 30
GENERATIONS = 20
ELITES = 6  # survivants/élites
TOURNAMENT_K = 4

SIM_DT = 1 / 240
SIM_DURATION_S = 25.0
MAX_STEPS = int(SIM_DURATION_S / SIM_DT)

N_EVAL = 3                # moyenne sur seeds
TORQUE_LIMIT = 2.0e6      # saturation couples

# -----------------------------
# Fitness / constraints
# -----------------------------
MAX_LANDING_SPEED = 8.0
MAX_TILT = 0.5

# NOUVEAU: anti-glissade (comme ton camarade)
MAX_SLIDE_SPEED = 2.0          # m/s au contact
SLIDE_PENALTY_GAIN = 400.0     # points par (m/s) au-dessus du seuil

# -----------------------------
# PARAMS OPTIMISÉS (LOG-SPACE)
# genome = [log10_kp, log10_kd, pip_h, pip_l]
# -----------------------------
BOUNDS = {
    "log_kp": [4.70, 5.85],    # ~ 50k -> 708k
    "log_kd": [4.00, 5.18],    # ~ 10k -> 151k
    "pip_h":  [0.001, 0.1],
    "pip_l":  [0.005, 0.2],
}

# Mutation (additive)
MUT_RATE = 0.35
SIGMA_LOG_KP = 0.08   # ~ multiplicatif moyen 10^0.08 ~ x1.20
SIGMA_LOG_KD = 0.10
SIGMA_PIP = 0.06      # variation relative pour pip_*


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def decode_genome(genome):
    """Convertit genome log-space -> paramètres réels pour FlightComputer."""
    log_kp, log_kd, pip_h, pip_l = genome
    kp = 10 ** log_kp
    kd = 10 ** log_kd
    return float(kp), float(kd), float(pip_h), float(pip_l)


def spawn_xy_on_circle(radius, rng: random.Random):
    a = rng.uniform(0, 2 * np.pi)
    return radius * np.cos(a), radius * np.sin(a)


def get_fitness_from_real_params(params, physics_client, difficulty_radius, seed=0):
    rng = random.Random(seed)

    # 1) setup sim
    p.resetSimulation(physicsClientId=physics_client)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(SIM_DT, physicsClientId=physics_client)

    plane_id = p.loadURDF("plane.urdf", physicsClientId=physics_client)

    cfg = RocketParams()
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=cfg.W / 2, height=cfg.H, physicsClientId=physics_client)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=cfg.W / 2, length=cfg.H,
                              rgbaColor=[1, 1, 1, 1], physicsClientId=physics_client)

    sx, sy = spawn_xy_on_circle(difficulty_radius, rng)
    rocket_id = p.createMultiBody(
        baseMass=cfg.mass_vide + cfg.mass_fuel,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[sx, sy, 300],
        physicsClientId=physics_client
    )

    p.changeDynamics(rocket_id, -1, lateralFriction=1.0, spinningFriction=0.1, physicsClientId=physics_client)

    computer = FlightComputer(*params)

    landed = False
    crashed = False
    fail_reason = ""

    # NEW: pour score glissade même après break
    v_horiz_contact = 0.0
    v_impact_contact = 0.0
    tilt_contact = 0.0

    # 2) loop
    for _ in range(MAX_STEPS):
        pos, orn = p.getBasePositionAndOrientation(rocket_id, physicsClientId=physics_client)
        lin_vel, ang_vel = p.getBaseVelocity(rocket_id, physicsClientId=physics_client)
        euler = p.getEulerFromQuaternion(orn)

        contacts = p.getContactPoints(bodyA=rocket_id, bodyB=plane_id, physicsClientId=physics_client)
        if len(contacts) > 0:
            tilt = max(abs(euler[0]), abs(euler[1]))
            v_impact = abs(lin_vel[2])
            v_horiz = float(np.sqrt(lin_vel[0] ** 2 + lin_vel[1] ** 2))

            tilt_contact = float(tilt)
            v_impact_contact = float(v_impact)
            v_horiz_contact = float(v_horiz)

            # 1) Angle
            if tilt > MAX_TILT:
                crashed = True
                fail_reason = "TROP PENCHÉ"

            # 2) Vitesse verticale
            elif v_impact > MAX_LANDING_SPEED:
                crashed = True
                fail_reason = f"VITESSE EXCESSIVE ({v_impact:.1f} m/s)"

            # 3) NOUVEAU: glissade horizontale
            elif v_horiz > MAX_SLIDE_SPEED:
                crashed = True
                fail_reason = f"GLISSADE ({v_horiz:.1f} m/s)"

            else:
                landed = True

            break

        # kill switch
        if pos[2] > 600 or pos[2] < -20 or abs(pos[0]) > 250 or abs(pos[1]) > 250:
            crashed = True
            fail_reason = "HORS LIMITES"
            break

        thrust, torques = computer.update(pos, lin_vel, euler, ang_vel, SIM_DT)

        torques = np.clip(np.asarray(torques, dtype=float), -TORQUE_LIMIT, TORQUE_LIMIT)

        p.changeDynamics(rocket_id, -1, mass=computer.current_mass, physicsClientId=physics_client)

        if thrust > 0:
            R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            p.applyExternalForce(
                rocket_id, -1, R @ [0, 0, thrust],
                [0, 0, -cfg.H / 2],
                p.LINK_FRAME,
                physicsClientId=physics_client
            )

        p.applyExternalTorque(rocket_id, -1, torques, p.WORLD_FRAME, physicsClientId=physics_client)
        p.stepSimulation(physicsClientId=physics_client)

    # 3) score
    pos, orn = p.getBasePositionAndOrientation(rocket_id, physicsClientId=physics_client)
    lin_vel, _ = p.getBaseVelocity(rocket_id, physicsClientId=physics_client)
    euler = p.getEulerFromQuaternion(orn)

    dist_center = float(np.sqrt(pos[0] ** 2 + pos[1] ** 2))
    tilt_final = float(max(abs(euler[0]), abs(euler[1])))
    vz_final = float(abs(lin_vel[2]))
    v_horiz_final = float(np.sqrt(lin_vel[0] ** 2 + lin_vel[1] ** 2))

    fuel_remaining = float(computer.current_mass - cfg.mass_vide)
    fuel_remaining = max(0.0, min(fuel_remaining, float(cfg.mass_fuel)))
    fuel_pct = fuel_remaining / float(cfg.mass_fuel) if cfg.mass_fuel > 0 else 0.0

    score = 0.0

    # progress
    start_dist = float(difficulty_radius)
    progress = max(0.0, start_dist - dist_center)
    score += progress * 10.0

    # petit malus doux si fin d'épisode avec vitesse horizontale (aide convergence)
    score -= 2.0 * v_horiz_final

    if crashed:
        score -= 2000.0

        if fail_reason.startswith("VITESSE"):
            score -= max(0.0, (vz_final - MAX_LANDING_SPEED)) * 100.0

        if fail_reason == "TROP PENCHÉ":
            score -= tilt_final * 500.0

        # NOUVEAU: malus glissade (basé sur la vitesse au contact si disponible)
        if fail_reason.startswith("GLISSADE"):
            vref = v_horiz_contact if v_horiz_contact > 0 else v_horiz_final
            score -= max(0.0, (vref - MAX_SLIDE_SPEED)) * SLIDE_PENALTY_GAIN

    elif landed:
        score += 10000.0
        if dist_center < 10.0: score += 2000.0
        if dist_center < 5.0:  score += 3000.0
        if dist_center < 2.0:  score += 5000.0

        softness = (MAX_LANDING_SPEED - vz_final) / MAX_LANDING_SPEED
        softness = max(0.0, min(1.0, softness))
        score += softness * 4000.0
        score += fuel_pct * 5000.0

        # BONUS: landing propre -> bonus si quasi pas de glissade
        score += max(0.0, (MAX_SLIDE_SPEED - v_horiz_final)) * 300.0

    else:
        score -= 4000.0

    return score


def evaluate_genome(genome, client, difficulty, gen_index):
    params = decode_genome(genome)
    base_seed = 100000 * (gen_index + 1)
    vals = [get_fitness_from_real_params(params, client, difficulty, seed=base_seed + i) for i in range(N_EVAL)]
    return float(np.mean(vals))


def random_genome():
    return [
        random.uniform(*BOUNDS["log_kp"]),
        random.uniform(*BOUNDS["log_kd"]),
        random.uniform(*BOUNDS["pip_h"]),
        random.uniform(*BOUNDS["pip_l"]),
    ]


def tournament_select(scored, k=TOURNAMENT_K):
    competitors = random.sample(scored, k)
    competitors.sort(key=lambda x: x[0], reverse=True)
    return competitors[0][1]


def crossover(g1, g2):
    return [g1[i] if random.random() < 0.5 else g2[i] for i in range(4)]


def mutate(genome, boost=False):
    rate = MUT_RATE * (2.0 if boost else 1.0)

    g = list(genome)

    if random.random() < rate:
        g[0] += random.gauss(0.0, SIGMA_LOG_KP)
    if random.random() < rate:
        g[1] += random.gauss(0.0, SIGMA_LOG_KD)

    if random.random() < rate:
        g[2] *= (1.0 + random.gauss(0.0, SIGMA_PIP))
    if random.random() < rate:
        g[3] *= (1.0 + random.gauss(0.0, SIGMA_PIP))

    g[0] = clamp(g[0], *BOUNDS["log_kp"])
    g[1] = clamp(g[1], *BOUNDS["log_kd"])
    g[2] = clamp(g[2], *BOUNDS["pip_h"])
    g[3] = clamp(g[3], *BOUNDS["pip_l"])

    return g


def run_training():
    client = p.connect(p.DIRECT)
    print("--- DÉBUT ENTRAÎNEMENT (LOG-SPACE kp/kd + contactPoints + dt=1/240) ---")
    print(f"dt={SIM_DT} | max_steps={MAX_STEPS} | N_EVAL={N_EVAL}")

    population = [random_genome() for _ in range(POPULATION_SIZE)]

    global_best = None
    global_best_score = -1e18

    current_difficulty = 2.0
    max_difficulty = 1100
    stagnation = 0
    height_increment = max_difficulty / GENERATIONS

    for gen in range(GENERATIONS):
        scored = []
        for genome in population:
            s = evaluate_genome(genome, client, current_difficulty, gen_index=gen)
            scored.append((s, genome))

        scored.sort(key=lambda x: x[0], reverse=True)
        gen_best_score, gen_best = scored[0]

        if gen_best_score > global_best_score:
            global_best_score = gen_best_score
            global_best = list(gen_best)
            stagnation = 0
            kp, kd, pip_h, pip_l = decode_genome(global_best)
            print(f" >> RECORD! Score={global_best_score:.0f} | kp={kp:.0f} kd={kd:.0f} pip_h={pip_h:.4f} pip_l={pip_l:.4f}")
        else:
            stagnation += 1

        if gen_best_score > 8000:
            current_difficulty = min(max_difficulty, current_difficulty + height_increment)
            print(f" >> SUCCÈS ! Difficulté -> {current_difficulty:.1f}m")

        kp, kd, _, _ = decode_genome(gen_best)
        print(f"Gen {gen+1:03d} | Dist={current_difficulty:.1f}m | Score={gen_best_score:.0f} | kp={kp:.0f} kd={kd:.0f}")

        boost = False
        if stagnation > 10:
            boost = True
            stagnation = 0
            print(" !! BOOST MUTATION !!")

        elites = [g for _, g in scored[:ELITES]]
        new_pop = [list(e) for e in elites]

        if global_best is not None:
            new_pop[0] = list(global_best)

        while len(new_pop) < POPULATION_SIZE:
            p1 = tournament_select(scored, k=TOURNAMENT_K)
            p2 = tournament_select(scored, k=TOURNAMENT_K)
            child = crossover(p1, p2)
            child = mutate(child, boost=boost)
            new_pop.append(child)

        population = new_pop

    p.disconnect(client)
    return global_best if global_best is not None else gen_best


if __name__ == "__main__":
    best_genome = run_training()
    best_params = decode_genome(best_genome)

    print("\n--- TEST VISUEL FINAL ---")
    print(f"Meilleurs paramètres (réels) : {best_params}")
    time.sleep(1)

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    p.setTimeStep(SIM_DT)

    cfg = RocketParams()
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=cfg.W / 2, height=cfg.H)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=cfg.W / 2, length=cfg.H, rgbaColor=[0, 1, 0, 1])

    rocket_id = p.createMultiBody(
        baseMass=cfg.mass_vide + cfg.mass_fuel,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[40, -40, 300]
    )
    p.changeDynamics(rocket_id, -1, lateralFriction=1.0)

    computer = FlightComputer(*best_params)

    while p.isConnected():
        pos, orn = p.getBasePositionAndOrientation(rocket_id)
        lin_vel, ang_vel = p.getBaseVelocity(rocket_id)
        euler = p.getEulerFromQuaternion(orn)

        thrust, torques = computer.update(pos, lin_vel, euler, ang_vel, SIM_DT)
        torques = np.clip(np.asarray(torques, dtype=float), -TORQUE_LIMIT, TORQUE_LIMIT)

        p.changeDynamics(rocket_id, -1, mass=computer.current_mass)
        if thrust > 0:
            R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            p.applyExternalForce(rocket_id, -1, R @ [0, 0, thrust], [0, 0, -cfg.H / 2], p.LINK_FRAME)
        p.applyExternalTorque(rocket_id, -1, torques, p.WORLD_FRAME)

        p.resetDebugVisualizerCamera(cameraDistance=25, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=pos)

        p.stepSimulation()
        time.sleep(SIM_DT)
