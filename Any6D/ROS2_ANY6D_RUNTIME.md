# ROS2 Any6D Runtime

Questa architettura divide il runtime in due processi:

- `Isaac Sim` su Windows pubblica `rgb`, `depth`, `mask`, `camera_info` e una richiesta JSON.
- `Any6D` su Linux sottoscrive i topic, esegue la stima e pubblica `geometry_msgs/PoseStamped`.

## Topic

- `/any6d/rgb` -> `sensor_msgs/Image` (`rgb8`)
- `/any6d/depth` -> `sensor_msgs/Image` (`16UC1`, millimetri)
- `/any6d/mask` -> `sensor_msgs/Image` (`mono8`)
- `/any6d/camera_info` -> `sensor_msgs/CameraInfo`
- `/any6d/request` -> `std_msgs/String`
- `/any6d/pred_pose_json` -> `std_msgs/String`

## Request payload

Il messaggio JSON su `/any6d/request` contiene:

- `request_id`
- `object_name`
- `t_w_c`
- `object_frame_correction`
- nomi dei topic

La risposta JSON su `/any6d/pred_pose_json` contiene:

- `request_id`
- `status`
- `position`
- `orientation_xyzw`
- `mock`
- `message` se `status=error`

`Any6D` usa:

`T_W_O_pred_isaac = T_W_C * T_C_O_pred_any6d_ref * inv(T_O_isaac_to_any6d_ref)`

## Script

- Isaac client: `/mnt/c/isaac-sim/my_scripts/ycb_pick_suction_v3_ros2.py`
- Linux pose server: `/home/iacopo/cv_final/Any6D/ros2_any6d_pose_server.py`

## Avvio

Su Linux:

```bash
source /opt/ros/<distro>/setup.bash
python3 /home/iacopo/cv_final/Any6D/ros2_any6d_pose_server.py
```

Per un test di sola comunicazione senza Any6D:

```bash
source /opt/ros/<distro>/setup.bash
python3 /home/iacopo/cv_final/Any6D/ros2_any6d_pose_server.py --mock
```

Per usare davvero Any6D e caricare subito il modello dell'oggetto:

```bash
source /opt/ros/humble/setup.bash
python3 /home/iacopo/cv_final/Any6D/ros2_any6d_pose_server.py --preload-object 003_cracker_box
```

Se l'avvio reale e' corretto, il server stampa:

- disponibilita' CUDA
- nome GPU
- path ai pesi
- conferma di preload dell'estimatore

Su Windows Isaac Sim:

```bash
python.bat /mnt/c/isaac-sim/my_scripts/ycb_pick_suction_v3_ros2.py
```

## Prossimi passi

- allineare QoS e clock tra i due processi
- sostituire la mask oracle con una mask reale o con un servizio dedicato
- passare da `std_msgs/String` a un messaggio ROS2 custom quando il protocollo si stabilizza
