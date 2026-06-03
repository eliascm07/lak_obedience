# lak_control — Control de marcha bípeda para LAK Obedience

Control LIPM + Preview Control + IK-QP, sin ROS2, sin Reinforcement Learning.

## Estructura

```
walking_controller_kajita/
├── model.py              # Clase LakObedience (carga URDF con Pinocchio)
├── preview_control.py    # Planificador LIPM + control preview ZMP  [sin cambios]
├── foot.py               # Trayectorias Bézier de pies, secuencias  [sin cambios]
├── state_machine.py      # FSM de fases de marcha                   [sin cambios]
├── inverse_kinematic.py  # IK como QP (OSQP)                        [sin cambios]
├── simulation.py         # Interfaz MuJoCo
├── walk.py               # Script principal
└── requirements.txt
```

## Instalación

```bash
pip install -r requirements.txt
```

> **Nota sobre Pinocchio:** el paquete `pin` en PyPI instala Pinocchio vía cmeel.
> Si ya tienes Pinocchio de ROS2, puedes usarlo añadiendo su path:
> ```bash
> export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH
> ```

## Uso

```bash
# Solo ver info del modelo (sin simulación)
python3 walk.py --urdf src/lak_description/urdf/lak_obedience.urdf --info-only

# Simulación sin visor (más rápido)
python3 walk.py --urdf src/lak_description/urdf/lak_obedience.urdf

# Simulación con visor 3D
python3 walk.py --urdf src/lak_description/urdf/lak_obedience.urdf --render
```

## Parámetros clave a ajustar

En `walk.py`, clase `WalkParams`:

| Parámetro | Valor inicial | Qué hace |
|-----------|--------------|----------|
| `zc` | 0.22 m | Altura del CoM. Verificar con `--info-only` |
| `l_stride` | 0.03 m | Longitud de paso. Empezar pequeño |
| `t_ss` | 0.8 s | Duración single support |
| `foot_height` | 0.015 m | Altura máxima del pie swing |
| `w_com` | 10.0 | Peso IK — seguir CoM |
| `w_mf` | 50.0 | Peso IK — seguir pie swing |

## Flujo de ejecución

```
LakObedience (Pinocchio) 
    ↓ postura default
compute_steps_sequence → trayectorias de pies (Bézier)
compute_zmp_ref        → referencia ZMP completa
compute_preview_control_matrices → ganancias Gi, Gx, Gd (DARE)
    ↓ loop a 500 Hz
update_control         → target CoM (LIPM preview)
solve_inverse_kinematics → q_des (QP con OSQP)
MujocoSimulator.apply_position_control + step()
```

## Troubleshooting

**El robot cae inmediatamente:** Ajustar `zc` con `--info-only` y copiar el valor real del CoM.

**IK no converge:** Reducir `l_stride` y `foot_height`. Aumentar `mu` a 1e-3.

**Oscilaciones:** Reducir `w_mf` o aumentar `t_ss`.