import rosbag
import matplotlib.pyplot as plt
import numpy
import tf.transformations
import collections
import pandas

PoseData = collections.namedtuple("PoseData",
                                  ["position", "orientation", "time"])

DistanceControlData = collections.namedtuple("DistanceControlData", [
    "active", "thrust_min", "thrust_max", "thrust", "base_thrust", "distance",
    "distance_setpoint", "leader_path_index", "path_index", "time"
])

RssiData = collections.namedtuple(
    "RssiData", ["rssi", "rssi_dbm", "noise", "noise_dbm", "time"])

RxData = collections.namedtuple("RxFailData", [
    "dropped", "t_dropped", "received", "t_received", "total", "t_total",
    "drop_rate", "t_drop_rate"
])


class Evaluator(object):
    def __init__(self, bagfile: rosbag.Bag):
        self.bagfile: rosbag.Bag = bagfile
        self.t0 = self.bagfile.get_start_time()
        self.te = self.bagfile.get_end_time()

    def extract_pose(self, topic, t0=None):
        if t0 is None:
            t0 = self.t0
        msg_list = [x for x in self.bagfile.read_messages(topics=topic)]
        n = len(msg_list)
        p = numpy.zeros([n, 3], float)
        q = numpy.zeros([n, 4], float)
        t = numpy.zeros([n], float)
        for i, (_, msg, ros_time) in enumerate(msg_list):
            rot = msg.pose.orientation
            p[i, :] = [
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
            ]
            q[i, :] = [rot.x, rot.y, rot.z, rot.w]
            t[i] = ros_time.to_sec() - t0
        return PoseData(position=p, orientation=q, time=t)

    def extract_rxdata(self, topic, t0=None):
        if t0 is None:
            t0 = self.t0

        dropped = []
        t_dropped = []
        received = []
        t_received = []
        total = []
        t_total = []
        drop_rate = []
        t_drop_rate = []
        for _, msg, ros_time in self.bagfile.read_messages(topics=topic):
            t = ros_time.to_sec() - t0
            append_counter(total)
            t_total.append(t)
            if msg.rssi == 0:
                append_counter(dropped)
                t_dropped.append(t)
            else:
                append_counter(received)
                t_received.append(t)
            r = dropped[-1] / total[-1] if len(dropped) > 0 else 0
            drop_rate.append(r)
            t_drop_rate.append(t)
        dropped = numpy.array(dropped)
        t_dropped = numpy.array(t_dropped)
        received = numpy.array(received)
        t_received = numpy.array(t_received)
        total = numpy.array(total)
        t_total = numpy.array(t_total)
        drop_rate = numpy.array(drop_rate)
        t_drop_rate = numpy.array(t_drop_rate)
        return RxData(dropped=dropped,
                      t_dropped=t_dropped,
                      received=received,
                      t_received=t_received,
                      total=total,
                      t_total=t_total,
                      drop_rate=drop_rate,
                      t_drop_rate=t_drop_rate)

    def extract_rssi(self, topic, t0=None):
        if t0 is None:
            t0 = self.t0
        msg_list = [x for x in self.bagfile.read_messages(topics=topic)]
        n = len(msg_list)
        rssi = numpy.zeros([n], int)
        rssi_dbm = numpy.zeros([n], float)
        noise = numpy.zeros([n], int)
        noise_dbm = numpy.zeros([n], float)
        t = []

        for i, (_, msg, ros_time) in enumerate(msg_list):
            rssi[i] = msg.rssi
            rssi_dbm[i] = msg.rssi_dbm
            noise[i] = msg.noise
            noise_dbm[i] = msg.noise_dbm
            t[i] = ros_time.to_sec() - t0

        mask = rssi != 0
        rssi_dbm = rssi_dbm[mask]
        noise = noise[mask]
        noise_dbm = noise_dbm[mask]
        t = t[mask]

        return RssiData(rssi=rssi,
                        rssi_dbm=rssi_dbm,
                        noise=noise,
                        noise_dbm=noise_dbm,
                        time=t)

    def extract_distance_control(self, topic, t0=None):
        if t0 is None:
            t0 = self.t0
        msg_list = [x for x in self.bagfile.read_messages(topics=topic)]
        n = len(msg_list)
        active = numpy.zeros([n], float)
        thrust_min = numpy.zeros_like(active)
        thrust_max = numpy.zeros_like(active)
        thrust = numpy.zeros_like(active)
        base_thrust = numpy.zeros_like(active)
        distance = numpy.zeros_like(active)
        distance_setpoint = numpy.zeros_like(active)
        leader_path_index = numpy.zeros_like(active)
        path_index = numpy.zeros_like(active)
        t = numpy.zeros_like(active)

        for i, (_, msg, ros_time) in enumerate(msg_list):
            active[i] = msg.active
            thrust_min[i] = msg.thrust_min
            thrust_max[i] = msg.thrust_max
            thrust[i] = msg.thrust
            base_thrust[i] = msg.base_thrust
            distance[i] = msg.distance
            distance_setpoint[i] = msg.distance_setpoint
            leader_path_index[i] = msg.leader_path_index
            path_index[i] = msg.path_index
            t[i] = ros_time.to_sec() - t0

        return DistanceControlData(active=active,
                                   thrust_min=thrust_min,
                                   thrust_max=thrust_max,
                                   thrust=thrust,
                                   base_thrust=base_thrust,
                                   distance=distance,
                                   distance_setpoint=distance_setpoint,
                                   leader_path_index=leader_path_index,
                                   path_index=path_index,
                                   time=t)


def moving_average_n(data, N):
    return numpy.convolve(data, numpy.ones((N, )) / N, mode="valid")


def movin_average_t(data, time, T):
    s = pandas.Series(data, index=pandas.to_datetime(time, unit="s"))
    mean = s.rolling(window="{}s".format(T), center=True).mean()
    print(mean)
    return numpy.array(mean.values.astype(float)), numpy.array(
        mean.index.values.astype(float) * 0.000000001)


def resample(data, time, time_sample):
    resampled_data = numpy.interp(time_sample, time, data)
    return resampled_data


def limit_to_shorter(a, b):
    if len(a) < len(b):
        return a, b[:len(a)]
    else:
        return a[:len(b)], b


def crop_data(data, time, t0, t1):
    tmp = numpy.abs(time - t0)
    a = tmp.argmin()
    tmp = numpy.abs(time - t1)
    b = tmp.argmin()
    return data[a:b], time[a:b]


def append_counter(a, inc=False):
    if len(a) == 0:
        a.append(1)
    else:
        if inc:
            a.append(a[-1] + 1)
        else:
            a.append(a[-1])


# bag = rosbag.Bag("21-07-16/21-07-16-path-follower-RF.bag")
# pos_x = []
# pos_x_t = []
# pos_y = []
# pos_y_t = []
# antenna_angle = []
# antenna_angle_t = []
# antenna2_axis = []
# antenna2_axis_t = []
# antenna3_axis = []
# antenna3_axis_t = []
# pos = dict()
# pos2 = []
# pos2_t = []
# pos3 = []
# pos3_t = []

# for topic, msg, t in bag.read_messages(topics=[
#         "/uuv02/mavros/local_position/pose", "/uuv03/mavros/local_position/pose"
# ]):
#     q = [
#         msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
#         msg.pose.orientation.w
#     ]
#     qprime = tf.transformations.quaternion_conjugate(q)
#     z = [0, 0, 1, 0]
#     v = tf.transformations.quaternion_multiply(
#         tf.transformations.quaternion_multiply(q, z), qprime)[0:-1]
#     position = msg.pose.position
#     pos = [position.x, position.y]
#     if topic == "/uuv02/mavros/local_position/pose":
#         antenna2_axis.append(v)
#         antenna2_axis_t.append(t.to_sec())
#         pos2.append(pos)
#         pos2_t.append(t.to_sec())
#     if topic == "/uuv03/mavros/local_position/pose":
#         antenna3_axis.append(v)
#         antenna3_axis_t.append(t.to_sec())
#         pos3.append(pos)
#         pos3_t.append(t.to_sec())

# t0 = antenna2_axis_t[0]
# pos2_t = [x - t0 for x in pos2_t]
# pos3_t = [x - t0 for x in pos3_t]
# n2 = len(antenna2_axis)
# n3 = len(antenna3_axis)
# if n2 > n3:
#     antenna2_axis = antenna2_axis[:n3]
#     antenna2_axis_t = antenna2_axis_t[:n3]
#     pos2 = pos2[:n3]
#     pos2_t = pos2_t[:n3]

# elif n3 > n2:
#     antenna3_axis = antenna3_axis[:n2]
#     antenna3_axis_t = antenna3_axis_t[:n2]
#     pos3 = pos3[:n2]
#     pos3_t = pos3_t[:n2]

# true_distance = []
# true_distance_t = []
# for i in range(len(antenna2_axis)):
#     tmp1 = numpy.array(pos2[i])
#     tmp2 = numpy.array(pos3[i])
#     v = tmp1 - tmp2
#     true_distance.append(numpy.linalg.norm(v))
#     true_distance_t.append(pos2_t[i])
#     v1 = antenna2_axis[i]
#     v2 = antenna3_axis[i]
#     antenna_angle.append(
#         numpy.arccos(
#             numpy.dot(v1, v2) /
#             (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))))
#     antenna_angle_t.append(antenna2_axis_t[i] - t0)

# pos_x_t = [v - pos_x_t[0] for v in pos_x_t]
# pos_y_t = [v - pos_y_t[0] for v in pos_y_t]

# distance_control = dict()
# distance_control["distance"] = []
# distance_control["distance_t"] = []
# distance_control["thrust"] = []
# distance_control["thrust_t"] = []
# distance_control["distance_target"] = []
# distance_control["distance_target_t"] = []
# for _, msg, t in bag.read_messages(
#         topics="/uuv03/distance_controller/distance_debug"):
#     distance_control["distance"].append(msg.distance)
#     distance_control["distance_t"].append(t.to_sec() - t0)
#     distance_control["distance_target"].append(msg.distance_setpoint)
#     distance_control["distance_target_t"].append(t.to_sec() - t0)
#     distance_control["thrust"].append(msg.thrust)
#     distance_control["thrust_t"].append(t.to_sec() - t0)

# rf = dict()
# rf["2to3"] = dict()
# rf["3to2"] = dict()
# rf["2to0"] = dict()
# rf["3to0"] = dict()
# for conn in list(rf):
#     rf[conn]["dropped"] = []
#     rf[conn]["total"] = []
#     rf[conn]["received"] = []
#     rf[conn]["drop_rate"] = []
#     rf[conn]["rssi"] = []
#     rf[conn]["rssi_dbm"] = []
#     rf[conn]["noise_dbm"] = []
#     for x in list(rf[conn]):
#         rf[conn]["{}_t".format(x)] = []

# for topic, msg, t in bag.read_messages(topics=[
#         "/uuv02/multi_uuv_rssi_03", "/uuv03/multi_uuv_rssi_02",
#         "/multi_uuv_rssi_02", "/multi_uuv_rssi_03"
# ]):
#     if topic == "/uuv02/multi_uuv_rssi_03":
#         conn = "3to2"
#     elif topic == "/uuv03/multi_uuv_rssi_02":
#         conn = "2to3"
#     elif topic == "/multi_uuv_rssi_02":
#         conn = "2to0"
#     elif topic == "/multi_uuv_rssi_03":
#         conn = "3to0"
#     else:
#         continue
#     append_counter(rf[conn]["total"])
#     if msg.rssi == 0:
#         append_counter(rf[conn]["dropped"])
#     else:
#         append_counter(rf[conn]["received"])
#         rf[conn]["rssi"].append(msg.rssi)
#         rf[conn]["rssi_dbm"].append(msg.rssi_dbm)
#     try:
#         dropped = rf[conn]["dropped"][-1]
#     except IndexError:
#         dropped = 0
#     rf[conn]["drop_rate"].append(float(dropped) / rf[conn]["total"][-1])
#     rf[conn]["noise_dbm"].append(msg.noise_dbm)
#     for x in rf[conn]:
#         if x[-1] != "t":
#             key = "{}_t".format(x)
#             if len(rf[conn][key]) != len(rf[conn][x]):
#                 rf[conn][key].append(t.to_sec() - t0)

# plt.figure()
# plt.plot(antenna_angle_t, antenna_angle)
# plt.grid(True)
# plt.twinx()
# N = 7
# conn = "3to2"
# plt.plot(
#     moving_average(rf[conn]["rssi_dbm_t"], N),
#     moving_average(rf[conn]["rssi_dbm"], N),
# )
# conn = "2to3"
# plt.plot(
#     moving_average(rf[conn]["rssi_dbm_t"], N),
#     moving_average(rf[conn]["rssi_dbm"], N),
# )

# plt.figure()
# N = 7
# conn = "3to2"
# plt.plot(
#     moving_average(rf[conn]["rssi_dbm_t"], N),
#     moving_average(rf[conn]["rssi_dbm"], N),
# )
# conn = "2to3"
# plt.plot(
#     moving_average(rf[conn]["rssi_dbm_t"], N),
#     moving_average(rf[conn]["rssi_dbm"], N),
# )
# plt.grid(True)
# plt.twinx()
# plt.plot(distance_control["distance_t"],
#          distance_control["distance"],
#          label="Distance",
#          linestyle="dashed",
#          color="tab:green")
# plt.plot(true_distance_t, true_distance, color="lime")

# plt.figure()
# plt.plot(distance_control["distance_target_t"],
#          distance_control["distance_target"],
#          label="Target Distance")
# plt.plot(distance_control["distance_t"],
#          distance_control["distance"],
#          label="Distance")
# plt.plot(distance_control["thrust_t"],
#          distance_control["thrust"],
#          label="Thrust")

# plt.legend()
