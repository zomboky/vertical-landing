import numpy as np


def wrap_pi(a: float) -> float:
    """Ramène un angle dans [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class RocketParams:
    """Paramètres physiques"""
    def __init__(self):
        self.mass_vide = 1500.0
        self.mass_fuel = 400.0
        self.g = 9.81
        self.nb_moteurs = 9
        self.F_max = 4600 * self.nb_moteurs
        self.H = 13.0
        self.W = 1.2

        # Hypothèses "contrôle" (tu peux ajuster)
        self.isp = 300.0              # s (utilisé pour dm/dt)
        self.thrust_margin = 0.90     # marge sur Fmax (éviter saturation permanente)

        # Limites de commande (stabilité)
        self.max_tilt_cmd_high = 0.25  # rad (~14°) haute altitude
        self.max_tilt_cmd_low = 0.50   # rad (~28°) basse altitude

        # Gains internes (suicide burn)
        self.kp_vz = 3500.0           # (N per m/s) plus doux que 5000
        self.kd_vz = 1200.0           # damping sur d(vz) approx
        self.thrust_rate = 80000.0    # N/s max de variation (anti “bang-bang”)


class FlightComputer:
    """
    Contrôleur :
    - Suicide burn (vertical speed shaping) avec limitation de rate sur thrust
    - PIP guidance (target pitch/roll) avec gain schedulé et saturation progressive
    - Attitude PD avec wrap angle et saturation torques
    """
    def __init__(self, kp_rot=60000, kd_rot=12000, pip_high=0.01, pip_low=0.03):
        self.cfg = RocketParams()
        self.current_mass = self.cfg.mass_vide + self.cfg.mass_fuel

        # --- PARAMS optimisables ---
        self.kp_rot = float(kp_rot)
        self.kd_rot = float(kd_rot)
        self.pip_gain_high = float(pip_high)
        self.pip_gain_low = float(pip_low)

        # --- états internes ---
        self.burn_active = False
        self.prev_vz = 0.0
        self.prev_thrust = 0.0

        # petit filtre sur ang_vel (optionnel)
        self._ang_vel_filt = np.zeros(3, dtype=float)
        self._ang_vel_alpha = 0.35  # 0=pas de filtre, 1=filtre très fort

        # torques limits (cohérent avec training)
        self.torque_limit = 2.0e6

    def update(self, pos, vel, euler, ang_vel, dt):
        z = float(pos[2])
        vz = float(vel[2])

        thrust_cmd = self._compute_suicide_burn(z, vz, dt)

        target_pitch, target_roll = self._compute_pip_guidance(
            pos=np.asarray(pos, dtype=float),
            vel=np.asarray(vel, dtype=float),
            z=z,
            vz=vz
        )

        # filtre ang_vel
        ang_vel = np.asarray(ang_vel, dtype=float)
        self._ang_vel_filt = (1.0 - self._ang_vel_alpha) * self._ang_vel_filt + self._ang_vel_alpha * ang_vel

        torque_x, torque_y, torque_z = self._compute_attitude_control(
            target_pitch=target_pitch,
            target_roll=target_roll,
            euler=np.asarray(euler, dtype=float),
            ang_vel=self._ang_vel_filt
        )

        # Consommation fuel (m_dot = T / (g0 * Isp))
        if thrust_cmd > 0.0:
            dm = (thrust_cmd / (self.cfg.g * self.cfg.isp)) * dt
            self.current_mass = max(self.cfg.mass_vide, self.current_mass - dm)

        return thrust_cmd, [torque_x, torque_y, torque_z]

    def _compute_suicide_burn(self, z, vz, dt):
        """
        Version plus stable:
        - calc une altitude d'ignition (comme toi)
        - suit une consigne vz(z) plus progressive
        - limite la vitesse de variation du thrust (anti bang-bang)
        """
        g = self.cfg.g
        m = self.current_mass
        Fmax = self.cfg.F_max * self.cfg.thrust_margin

        weight = m * g
        a_max = (Fmax / m) - g
        if a_max <= 0.0:
            return self.cfg.F_max  # pas contrôlable

        # Distance de freinage verticale si on freine à a_max
        braking_dist = (vz ** 2) / (2.0 * a_max)
        ignition_alt = braking_dist + 8.0  # un peu plus conservateur que 5m

        if (z < ignition_alt) and (vz < -0.5):
            self.burn_active = True

        # coupure finale proche du sol
        if z < 0.6 and abs(vz) < 0.8:
            self.burn_active = False
            self.prev_thrust = 0.0
            return 0.0

        if not self.burn_active:
            self.prev_thrust = 0.0
            return 0.0

        # --- consigne vz(z) progressive ---
        # plus z est bas, plus on veut être lent.
        # exemple: v_target ~ -sqrt(2 a_max z) - offset, mais adouci
        v_free = -np.sqrt(max(0.0, 2.0 * a_max * max(z, 0.0)))
        v_target = 0.75 * v_free - 1.0  # adoucissement + léger offset

        # erreur de vitesse + terme d'amortissement sur dvz
        dvz = (vz - self.prev_vz) / max(dt, 1e-6)
        self.prev_vz = vz

        err_v = (v_target - vz)

        thrust_raw = weight + self.cfg.kp_vz * err_v - self.cfg.kd_vz * dvz

        # clamp thrust
        thrust_raw = float(np.clip(thrust_raw, 0.0, Fmax))

        # limitation de rate (N/s)
        max_delta = self.cfg.thrust_rate * dt
        thrust_cmd = float(np.clip(thrust_raw, self.prev_thrust - max_delta, self.prev_thrust + max_delta))
        self.prev_thrust = thrust_cmd
        return thrust_cmd

    def _compute_pip_guidance(self, pos, vel, z, vz):
        """
        Mix PIP (haut) + terminal guidance anti-glissade (bas)
        - haut: PIP classique
        - bas (<20-30m): annule vx/vy d'abord (comme ton camarade)
        """
        if z < 1.0:
            return 0.0, 0.0

        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)

        # --- Terminal mode (anti-glissade) ---
        # Sous 20m: priorité absolue -> vx=vy=0
        if z < 20.0:
            target_vx, target_vy = 0.0, 0.0
            kp_lat = 0.35  # fort pour figer
            err_vx = target_vx - vel[0]
            err_vy = target_vy - vel[1]

            # on convertit "acc lat souhaitée" -> inclinaison (petits angles)
            ax_cmd = kp_lat * err_vx
            ay_cmd = kp_lat * err_vy

            max_tilt = 0.15  # très limité proche du sol
            target_pitch = float(np.clip(+ax_cmd, -max_tilt, max_tilt))
            target_roll = float(np.clip(-ay_cmd, -max_tilt, max_tilt))
            return target_pitch, target_roll

        # --- Mid/High altitude: mix vitesse + PIP ---
        # 1) Guidance vitesse (ramener vers centre en tau secondes)
        tau = 6.0 if z > 60.0 else 4.0
        target_vx = -pos[0] / tau
        target_vy = -pos[1] / tau
        limit_v = 35.0
        target_vx = float(np.clip(target_vx, -limit_v, limit_v))
        target_vy = float(np.clip(target_vy, -limit_v, limit_v))

        kp_lat = 0.18 if z > 60.0 else 0.25
        err_vx = target_vx - vel[0]
        err_vy = target_vy - vel[1]
        ax_cmd = kp_lat * err_vx
        ay_cmd = kp_lat * err_vy

        # 2) PIP (pour anticiper)
        t_go = z / max(abs(vz), 6.0)
        t_go = float(np.clip(t_go, 0.8, 12.0))
        pip_x = pos[0] + vel[0] * t_go
        pip_y = pos[1] + vel[1] * t_go

        # gain schedulé comme avant
        z_hi, z_lo = 250.0, 50.0
        if z >= z_hi:
            gain = self.pip_gain_high
        elif z <= z_lo:
            gain = self.pip_gain_low
        else:
            a = (z - z_lo) / (z_hi - z_lo)
            gain = (1.0 - a) * self.pip_gain_low + a * self.pip_gain_high

        pip_pitch = pip_x * gain
        pip_roll = -pip_y * gain

        # 3) Mix (plus on descend, plus on privilégie la vitesse)
        # z=200 => w_vel~0.2 ; z=30 => w_vel~0.8
        w_vel = float(np.clip((200.0 - z) / (200.0 - 30.0), 0.2, 0.8))

        target_pitch = (1.0 - w_vel) * pip_pitch + w_vel * ax_cmd
        target_roll = (1.0 - w_vel) * pip_roll + w_vel * (-ay_cmd)

        # limites d'angle progressives (comme ton fichier actuel)
        z_hi2, z_lo2 = 200.0, 30.0
        if z >= z_hi2:
            max_tilt = self.cfg.max_tilt_cmd_high
        elif z <= z_lo2:
            max_tilt = self.cfg.max_tilt_cmd_low
        else:
            a = (z - z_lo2) / (z_hi2 - z_lo2)
            max_tilt = (1.0 - a) * self.cfg.max_tilt_cmd_low + a * self.cfg.max_tilt_cmd_high

        target_pitch = float(np.clip(target_pitch, -max_tilt, max_tilt))
        target_roll = float(np.clip(target_roll, -max_tilt, max_tilt))
        return target_pitch, target_roll

    def _compute_attitude_control(self, target_pitch, target_roll, euler, ang_vel):
        """
        PD attitude amélioré:
        - wrap angles (évite discontinuité yaw)
        - deadband sur petite erreur
        - saturation des couples
        """
        # erreurs (avec wrap)
        err_roll = wrap_pi(target_roll - float(euler[0]))
        err_pitch = wrap_pi(target_pitch - float(euler[1]))
        err_yaw = wrap_pi(0.0 - float(euler[2]))

        # deadband (évite tremblement)
        def deadband(x, eps=0.01):
            return 0.0 if abs(x) < eps else x

        err_roll = deadband(err_roll)
        err_pitch = deadband(err_pitch)
        err_yaw = deadband(err_yaw)

        # PD (optimisé par GA)
        tx = self.kp_rot * err_roll - self.kd_rot * float(ang_vel[0])
        ty = self.kp_rot * err_pitch - self.kd_rot * float(ang_vel[1])
        tz = self.kp_rot * err_yaw - self.kd_rot * float(ang_vel[2])

        # saturation finale
        tx = float(np.clip(tx, -self.torque_limit, self.torque_limit))
        ty = float(np.clip(ty, -self.torque_limit, self.torque_limit))
        tz = float(np.clip(tz, -self.torque_limit, self.torque_limit))
        return tx, ty, tz
