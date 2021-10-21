import rosbag
import matplotlib.pyplot as plt
import numpy
import os
import matplotlib
import pandas as pd

import rosbag_eval as be

BASE_DIR = os.path.expanduser("~/uuv/catkin_ws/bagfile_evaluation")
SUB_DIR = "leader-follower/sim"
FILE_NAME = "sim-circle-0.5-thrust.bag"
BAGFILE = os.path.join(BASE_DIR, SUB_DIR, FILE_NAME)

CMAP = matplotlib.cm.get_cmap("tab10")

RATE_MAX = 7.46


def plot_path_distance(e, t0=None, t1=None):
    plt.figure()
    dcontrol = e.extract_distance_control(
        "/uuv03/distance_controller/distance_debug")
    if t0 is not None and t1 is not None:
        path_dist, t_path_dist = be.crop_data(dcontrol.distance, dcontrol.time,
                                              t0, t1)
        dist_target, t_dist_target = be.crop_data(dcontrol.distance_setpoint,
                                                  dcontrol.time, t0, t1)
    else:
        path_dist = dcontrol.distance
        t_path_dist = dcontrol.time
        dist_target = dcontrol.distance_setpoint
        t_dist_target = dcontrol.time
    plt.plot(t_path_dist, path_dist, label="Distanz")
    plt.plot(t_dist_target, dist_target, label="Zieldistanz")
    d = numpy.array([t_path_dist, path_dist, dist_target], dtype=float)
    out = pd.DataFrame(numpy.transpose(d),
                       columns=["Zeit", "Istwert", "Sollwert"])
    out.to_csv("distance_setpoint.csv", sep=",")


def plot_path_distance_and_x_coord(e):
    plt.figure()
    dcontrol = e.extract_distance_control(
        "/uuv03/distance_controller/distance_debug")
    pose_leader = e.extract_pose("/uuv02/ground_truth/state")
    pose_follower = e.extract_pose("/uuv03/ground_truth/state")
    toffset = 0
    tspan = 60
    tend = toffset + tspan

    data, t = be.crop_data(dcontrol.distance, dcontrol.time, toffset, tend)
    plt.plot(t - toffset, data, label="Pfadabstand A zu B", color=CMAP(0))
    plt.xlabel("Zeit [s]")
    plt.ylabel("Abstand [m]")
    d = numpy.array([t, data], dtype=float)
    out = pd.DataFrame(numpy.transpose(d), columns=["Zeit", "Pfadabstand"])
    out.to_csv("path_distance_fig2.csv", sep=",")

    data, t = be.crop_data(pose_follower.position[:, 0], pose_follower.time,
                           toffset, tend)
    plt.ylabel("$x$-Koordinate [m]")
    plt.plot(t - toffset, data, label="$x$-Koordinate", color=CMAP(1))
    d = numpy.array([t, data], dtype=float)
    out = pd.DataFrame(numpy.transpose(d), columns=["Zeit", "x"])
    out.to_csv("x_coord_fig2.csv", sep=",")
    plt.legend()
    plt.xlim((0, tspan))


def plot_xy(e):
    plt.figure()
    t0 = 0
    t1 = 60
    tspan = t1 - t0
    pose_leader = e.extract_pose("/uuv02/ground_truth/state")
    pose_follower = e.extract_pose("/uuv03/ground_truth/state")
    path_target = e.extract_path_info("/uuv03/path_follower/target")
    pos_leader = pose_leader.position
    pos_follower = pose_follower.position
    t_leader = pose_leader.time
    t_follower = pose_follower.time
    pos_target, t_target = be.crop_data(path_target.target_position,
                                        path_target.time, t0, t1)
    pos_leader, t_leader = be.crop_data(pos_leader, t_leader, t0, t1)
    t_leader -= t0
    pos_follower, t_follower = be.crop_data(pos_follower, t_follower, t0, t1)
    t_follower -= t0
    plt.plot(pos_leader[:, 0], pos_leader[:, 1], label="Fahrzeug A")
    plt.plot(pos_follower[:, 0], pos_follower[:, 1], label="Fahrzeug B")
    plt.plot(pos_target[:, 0], pos_target[:, 1], "k", label="Zielpfad")
    plt.axis("equal")
    plt.legend()
    plt.xlabel("$x$-Koordinate [m]")
    plt.ylabel("$y$-Koordinate [m]")
    d = numpy.array([pos_leader[:, 0], pos_leader[:, 1]], dtype=float)
    out = pd.DataFrame(numpy.transpose(d), columns=["x", "y"])
    out.to_csv("xy_leader.csv")
    d = numpy.array([pos_follower[:, 0], pos_follower[:, 1]], dtype=float)
    out = pd.DataFrame(numpy.transpose(d), columns=["x", "y"])
    out.to_csv("xy_follower.csv")
    d = numpy.array([pos_target[:, 0], pos_target[:, 1]], dtype=float)
    out = pd.DataFrame(numpy.transpose(d), columns=["x", "y"])
    out.to_csv("xy_path.csv")


def main():
    bagfile = rosbag.Bag(BAGFILE)
    e = be.Evaluator(bagfile)
    plot_path_distance(e)
    # Was sieht man? Die Regelgröße "Pfaddistanz" schwankt periodisch, abhängig
    # davon, wo sich das Fahrzeug auf der 8 befindet. -> 2 Mögliche Ursachen:
    # Fahrzeugdynamik abhängig von Kurvenfahrt (an den Extrema der x-Position
    # lenkt es besonders stark ein) wird das Fahrzeug schneller bzw. langsamer.
    # Außerdem: Vision hat besonders bei Rotation Schwierigkeiten, die Position
    # richtig zu bestimmen -> Hier entspricht eine Änderung der Position nicht
    # zwangsweise der Wahrheit. Bildverzerrung hat beim gieren großen Einfluss
    # auf Positionsfehler.

    plot_path_distance_and_x_coord(e)

    # xy Plot der Fahrzeuge -> gut wiederholbar, aber verzerrt
    plot_xy(e)

    plt.show()


if __name__ == "__main__":
    main()
