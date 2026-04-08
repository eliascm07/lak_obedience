# 🤖 LAK Obedience - Bidep Robot Project

Este repositorio contiene un workspace de ROS 2 para el modelo inicial en urdf del robot.

---

## 📦 Requisitos

* ROS 2 (ej: Humble, Jazzy)
* Python 3
* `colcon`

Instalar dependencias:

```bash
sudo apt update
sudo apt install python3-colcon-common-extensions
```

---

## 🚀 Clonar el repositorio

```bash
git clone https://github.com/eliascm07/lak_obedience.git
cd lak_obedience
```

---

## 🧱 Compilar el workspace

```bash
# Clonar el repo dentro de src (Verificar versión de ROS2)
source /opt/ros/jazzy/setup.bash
colcon build
```

---

## 🔌 Cargar el entorno

```bash
source install/setup.bash
```

---

## ▶️ Ejecutar el proyecto

```bash
ros2 launch lak_bringup display.launch.py
```

---

## 📁 Estructura del proyecto

```bash
src/
 ├── lak_bringup/
 ├── otros_paquetes/
```

---


## ⚠️ Notas

* No se incluyen las carpetas `build/`, `install/`, `log/` ya que se generan automáticamente.
* Asegúrate de tener correctamente instalado ROS 2 antes de compilar.

---

## 👨‍💻 Autores

Tu Nombre
