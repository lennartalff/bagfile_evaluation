import rosbag
import matplotlib.pyplot as plt
import numpy
import tf.transformations
import collections
import os
import matplotlib

import rosbag_eval as be

BASE_DIR = os.path.expanduser("~/uuv/catkin_ws/eval_bagfiles")
SUB_DIR = "leader-follower/21-07-20"
FILE_NAME = "21-07-20-path-follower-RF-P-gains.bag"
BAGFILE = os.path.join(BASE_DIR, SUB_DIR, FILE_NAME)

CMAP = matplotlib.cm.get_cmap("tab10")


def main():
    bagfile = rosbag.Bag(BAGFILE)
    e = be.Evaluator(bagfile)
    pose_leader = e.extract_pose("/uuv02/mavros/local_position/pose")
    pose_follower = e.extract_pose("/uuv03/mavros/local_position/pose")
    dcontrol = e.extract_distance_control(
        "/uuv03/distance_controller/distance_debug")

    plt.plot(dcontrol.time, dcontrol.distance, label="Distanz")
    plt.plot(dcontrol.time, dcontrol.distance_setpoint, label="Zieldistanz")

    # Was sieht man? Die Regelgröße "Pfaddistanz" schwankt periodisch, abhängig
    # davon, wo sich das Fahrzeug auf der 8 befindet. -> 2 Mögliche Ursachen:
    # Fahrzeugdynamik abhängig von Kurvenfahrt (an den Extrema der x-Position
    # lenkt es besonders stark ein) wird das Fahrzeug schneller bzw. langsamer.
    # Außerdem: Vision hat besonders bei Rotation Schwierigkeiten, die Position
    # richtig zu bestimmen -> Hier entspricht eine Änderung der Position nicht
    # zwangsweise der Wahrheit. Bildverzerrung hat beim gieren großen Einfluss
    # auf Positionsfehler.
    toffset = 130
    tspan = 70
    tend = toffset + tspan
    plt.figure()

    data, t = be.crop_data(dcontrol.distance, dcontrol.time, toffset, tend)
    plt.plot(t - toffset, data, label="Pfadabstand A zu B", color=CMAP(0))
    plt.xlabel("Zeit [s]")
    plt.ylabel("Abstand [m]")

    data, t = be.crop_data(pose_follower.position[:, 0], pose_follower.time,
                           toffset, tend)
    plt.ylabel("$x$-Koordinate [m]")
    plt.plot(t - toffset, data, label="$x$-Koordinate", color=CMAP(1))

    plt.legend()
    plt.xlim((0, tspan))

    # plt.figure()

    # pose_leader = e.extract_pose("/uuv02/mavros/local_position/pose")
    # pose_follower = e.extract_pose("/uuv03/mavros/local_position/pose")
    # rssi_2to3 = e.extract_rssi("/uuv03/multi_uuv_rssi_02")
    # pos_a = pose_leader.position
    # pos_b = pose_follower.position
    # t_a = pose_leader.time
    # t_b = pose_follower.time
    # pos_a, pos_b = be.limit_to_shorter(pos_a, pos_b)
    # t_a, t_b = be.limit_to_shorter(t_a, t_b)

    # d_vec = pos_a - pos_b
    # d = numpy.sum(d_vec**2, axis=-1)**(0.5)
    # plt.plot(t_a, d, rssi_2to3.time, rssi_2to3.rssi_dbm)

    plt.show()


if __name__ == "__main__":
    main()
