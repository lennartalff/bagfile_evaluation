import rosbag
import matplotlib.pyplot as plt
import numpy
import os
import matplotlib
import pandas as pd

import rosbag_eval as be

BASE_DIR = os.path.expanduser("~/uuv/catkin_ws/bagfile_evaluation")
SUB_DIR = "leader-follower/21-07-20"
FILE_NAME = "21-07-20-path-follower-RF-P-gains.bag"
BAGFILE = os.path.join(BASE_DIR, SUB_DIR, FILE_NAME)

CMAP = matplotlib.cm.get_cmap("tab10")

RATE_MAX = 7.72


def main():
    bagfile = rosbag.Bag(BAGFILE)
    e = be.Evaluator(bagfile)
    pose_leader = e.extract_pose("/uuv02/mavros/local_position/pose")
    pose_follower = e.extract_pose("/uuv03/mavros/local_position/pose")
    dcontrol = e.extract_distance_control(
        "/uuv03/distance_controller/distance_debug")

    plt.plot(dcontrol.time, dcontrol.distance, label="Distanz")
    plt.plot(dcontrol.time, dcontrol.distance_setpoint, label="Zieldistanz")
    d = numpy.array(
        [dcontrol.time, dcontrol.distance, dcontrol.distance_setpoint],
        dtype=float)
    out = pd.DataFrame(numpy.transpose(d),
                       columns=["Zeit", "Istwert", "Sollwert"])
    out.to_csv("distance_setpoint.csv", sep=",")
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

    # Signalstärke und Entfernung korrelieren sehr gut miteinander.
    plt.figure()

    pose_leader = e.extract_pose("/uuv02/mavros/local_position/pose")
    pose_follower = e.extract_pose("/uuv03/mavros/local_position/pose")
    rssi_2to3 = e.extract_rssi("/uuv03/multi_uuv_rssi_02")
    pos_a = pose_leader.position
    pos_b = pose_follower.position
    t_a = pose_leader.time
    t_b = pose_follower.time
    pos_a, pos_b = be.limit_to_shorter(pos_a, pos_b)
    t_a, t_b = be.limit_to_shorter(t_a, t_b)

    d_vec = pos_a - pos_b
    d = numpy.sum(d_vec**2, axis=-1)**(0.5)
    d, t_a = be.crop_data(d, t_a, toffset, tend)
    t_a -= toffset
    rssi_dbm, t_rssi_dbm = be.crop_data(rssi_2to3.rssi_dbm, rssi_2to3.time,
                                        toffset, tend)
    N = 10
    t_rssi_dbm = be.moving_average_n(t_rssi_dbm, N) - toffset
    rssi_dbm = be.moving_average_n(rssi_dbm, N)
    plt.plot(t_rssi_dbm, rssi_dbm, "k--", label="Signalstärke A zu B")
    plt.legend()
    plt.xlabel("Zeit [s]")
    plt.ylabel("Signalstärke [dBm]")
    plt.twinx()
    plt.plot(t_a, d, color=CMAP(1), label="Abstand A und B")
    plt.ylabel("Abstand [m]")
    plt.legend()
    plt.xlim(0, tspan)

    plt.figure()
    rf2to3 = e.extract_rxdata("/uuv03/multi_uuv_rssi_02")
    rf3to2 = e.extract_rxdata("/uuv02/multi_uuv_rssi_03")
    plt.plot(rf2to3.t_drop_rate, rf2to3.drop_rate, label="A zu B")
    plt.plot(rf3to2.t_drop_rate, rf3to2.drop_rate, label="B zu A")
    plt.ylabel("Anteil verlorener Nachrichten")
    plt.xlabel("Zeit [s]")
    plt.legend()

    plt.figure()
    # xy Plot der Fahrzeuge -> gut wiederholbar, aber verzerrt
    t0 = 130
    t1 = 170
    tspan = t1 - t0
    pose_leader = e.extract_pose("/uuv02/mavros/local_position/pose")
    pose_follower = e.extract_pose("/uuv03/mavros/local_position/pose")
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

    plt.figure()
    t0 = 130
    t1 = 225
    tspan = t1 - t0
    dcontrol = e.extract_distance_control(
        "/uuv03/distance_controller/distance_debug")
    path_dist, t_path_dist = be.crop_data(dcontrol.distance, dcontrol.time, t0,
                                          t1)
    dist_target, t_dist_target = be.crop_data(dcontrol.distance_setpoint,
                                              dcontrol.time, t0, t1)
    thrust, t_thrust = be.crop_data(dcontrol.thrust, dcontrol.time, t0, t1)
    plt.subplot(2, 1, 1)
    plt.plot(t_path_dist - t0, path_dist, label="Pfaddistanz")
    plt.plot(t_dist_target - t0, dist_target, label="Zieldistanz")
    plt.ylabel("Distanz [m]")
    plt.xlabel("Zeit [s]")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel("Zeit [s]")
    plt.plot(t_thrust - t0, thrust, "k", label="ESC Signal")
    plt.ylabel("Normiertes ESC Eingangssignal")
    plt.legend()
    d = numpy.array([t_path_dist, path_dist, dist_target, thrust], dtype=float)
    out = pd.DataFrame(numpy.transpose(d),
                       columns=["Zeit", "Pfaddistanz", "Zieldistanz", "ESC"])
    out.to_csv("control_switch_off.csv")

    plt.figure()
    plt.subplot(2, 2, 1)
    noise2 = e.extract_rssi("/uuv02/multi_uuv_rssi_03")
    noise3 = e.extract_rssi("/uuv03/multi_uuv_rssi_02")
    rate3to2 = e.extract_rf_rate("/uuv02/multi_uuv_pose_03/rate")
    rate2to3 = e.extract_rf_rate("/uuv03/multi_uuv_pose_02/rate")
    t_t = rate2to3.time.copy()
    r_t = numpy.ones_like(t_t) * RATE_MAX
    plt.plot(rate2to3.time, rate2to3.rate, label="A zu B")
    plt.plot(t_t, r_t, label="Theoretisches Maximum")
    plt.ylabel("Paketrate [Hz]")
    plt.xlabel("Zeit [s]")
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(rate3to2.time, rate3to2.rate, label="B zu A")
    plt.plot(t_t, r_t, label="Theoretisches Maximum")
    plt.ylabel("Paketrate [Hz]")
    plt.xlabel("Zeit [s]")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    N = 10
    n2 = be.moving_average_n(noise2.noise_dbm, N)
    t2 = be.moving_average_n(noise2.time, N)
    n3 = be.moving_average_n(noise3.noise_dbm, N)
    t3 = be.moving_average_n(noise3.time, N)
    s2 = be.moving_average_n(noise2.rssi_dbm, N)
    s3 = be.moving_average_n(noise3.rssi_dbm, N)
    plt.plot(t2, n2, label="A")
    plt.plot(t3, n3, label="B")
    plt.ylabel("Rauschen [dBm]")
    plt.xlabel("Zeit [s]")
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(t2, s2, label="A")
    plt.plot(t3, s3, label="B", linestyle="dashed")
    plt.ylabel("Signalstärke [dBm]")
    plt.xlabel("Zeit [s]")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
