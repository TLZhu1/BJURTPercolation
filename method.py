import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import copy
from add_code import func_enhance_net_gen

# 研究线路范围
def all_line_station(line_station: list):
    new_line_station = []
    for i in line_station:
        if i[0] not in ['S052', 'S092', 'S101', 'S102', 'S190', 'S021']:
            new_line_station.append(i)
    return new_line_station

# 生成所有区间及其上下行字典
def fuc_section_gen(line_station: list):
    # section_all 变量代表所有的区间，为列表形式。section_dict 变量为字典，能够查询区间为上行还是下行
    section_all = []
    section_dict = {}
    # 该循环去除原始列表里空值元素
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x != None, line_station[i]))
    for i in line_station:
        if i[0] not in ['S052', 'S092', 'S101', 'S102', 'S190', 'S021']: # 研究线路范围
            if i[0] not in ['S020', 'S021', 'S100']: # 非环线
                t_list = i[1:]
                for j in range(len(t_list) - 1):
                    section_all.append((t_list[j], t_list[j + 1]))
                    section_dict[(t_list[j], t_list[j + 1])] = 0 # 0代表上行
                    section_all.append((t_list[j + 1], t_list[j]))
                    section_dict[(t_list[j + 1], t_list[j])] = 1 # 1代表下行
            elif i[0] in ['S020', 'S100']:
                t_list = i[1:]
                for j in range(len(t_list)):
                    section_all.append((t_list[j - 1], t_list[j]))
                    section_dict[(t_list[j - 1], t_list[j])] = 0 # 0代表上行
                    section_all.append((t_list[j], t_list[j - 1]))
                    section_dict[(t_list[j], t_list[j - 1])] = 1 # 1代表下行
            elif i[0] in ['S021']:
                t_list = [('北新桥', '东直门'), ('东直门', '北新桥'), ('东直门', '三元桥'), ('三元桥', '东直门'), ('三元桥', 'T3航站楼'), 
                ('T3航站楼', '三元桥'), ('T3航站楼', 'T2航站楼'), ('T2航站楼', 'T3航站楼'), ('T2航站楼', '三元桥'), ('三元桥', 'T2航站楼')]
                for j in range(len(t_list)):
                    if j%2 == 0:
                        section_all.append(t_list[j])
                        section_dict[t_list[j]] = 0
                    else:
                        section_all.append(t_list[j])
                        section_dict[t_list[j]] = 1
    return section_all, section_dict

# 给定区间，返回线路编号
def fuc_identify_line_id(section, line_station):
    result_id = []
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x!= None, line_station[i]))
    for i in line_station:
        if section[0] in i and section[1] in i:
            result_id.append(i[0])
    return result_id

# 给定车站，返回线路编号
def fuc_identify_line_id2(station: str, line_station: list):
    result_id = []
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x!= None, line_station[i]))
    for i in line_station:
        if station in i:
            result_id.append(i[0])
    return result_id

# 筛选出该条线路工作or非工作日列车时刻表（初筛）
def fuc_one_get_timetable(timetable: list, line_id: list, work_rest: int):
    result_timetable = []
    for i in line_id:
        for j in timetable:
            if i == j[0] and (work_rest == int(j[1]) or work_rest == 2):
                result_timetable.append(j)
    return result_timetable

# 给定区间与交路，返回该区间前面的区间
def fuc_pre_section(section: tuple, line_station: list, routing_num: int, routing: list, line_id: str):
    result_section = []
    # routing_section = ('x', 'y')
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x!= None, line_station[i]))
    for i in routing:
        if i[0] == line_id and int(i[1]) == routing_num:
            routing_section = (i[2].split(',')[0], i[3].split(',')[0])
    for i in line_station:
        ccc = i[0]
        cccc = i[1:]
        if ccc == line_id:
            b1 = cccc.index(routing_section[0])
            b2 = cccc.index(routing_section[1])
            if b1 < b2:
                temp_line = cccc[b1: b2 + 1]
            elif b1 > b2:
                temp_line = cccc[b2: b1 + 1]
    # -----------------------------------------
    if section[0] in temp_line and section[1] in temp_line:
        a1 = temp_line.index(section[0])
        a2 = temp_line.index(section[1])
    else:
        return []
    if a1 < a2:
        line_cut_list = temp_line[: a2 + 1]
    elif a1 > a2:
        line_cut_list = temp_line[a2 :]
    if len(line_cut_list) == 2:
        result_section.append((line_cut_list[0], line_cut_list[1]))
    else:
        for i in range(len(line_cut_list)):
            if i+1 <= len(line_cut_list)-1:
                result_section.append((line_cut_list[i], line_cut_list[i+1]))
    return result_section

# 给定线路编号，始发站与交路编号，判断上下行
def fuc_up_down(line_id: str, O_station: str, routing_num: int, routing: list, line_station: list):
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x!= None, line_station[i]))
    section_all, section_dict = fuc_section_gen(line_station)
    for i in routing:
        if i[0] == line_id and i[1] == routing_num:
            o_list = i[2].split(',')
            d_list = i[3].split(',')
            for j in line_station:
                if i[0] == line_id == j[0]:
                    os_s = o_list[o_list.index(O_station)]
                    ds_s = d_list[o_list.index(O_station)]
                    temp_list = j[1:]
                    a1 = temp_list.index(os_s)
                    a2 = temp_list.index(ds_s)
                    if a1 < a2:
                        example_section = (temp_list[0], temp_list[1])
                    else:
                        example_section = (temp_list[1], temp_list[0])
                    return section_dict[example_section]

# 车辆到达时间，给定区间，计算出当日所有车辆的到达时间
def get_train_time(section: tuple, timetable: list, work_rest: int, line_station: list, running_time: list, routing: list):
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x!= None, line_station[i]))
    result_time_list = []
    filter_timetable = fuc_one_get_timetable(timetable, fuc_identify_line_id(section, line_station), work_rest)
    for i in filter_timetable:
        line_id = i[0]
        o_station_name = i[2]
        hour = int(i[3])
        min_list = str(i[4]).split(',')
        for j in min_list:
            ax = j.split(';')
            if len(ax) == 1:
                mins = int(float(ax[0]))
                routing_num = 0
            elif len(ax) == 2:
                mins = int(float(ax[0]))
                routing_num = int(ax[1])
            section_list = fuc_pre_section(section, line_station, routing_num, routing, line_id)
            sum_second = 0
            if len(section_list) != 0:
                for m in section_list:
                    for n in running_time:
                        if m[0] in n and m[1] in n:
                            sum_second += int(n[2])
            else:
                continue
            mins += int(sum_second/60)
            remain_second = sum_second%60
            hour += int(mins/60)
            remain_min = mins%60
            timetable_routing_num = fuc_up_down(line_id, o_station_name, routing_num, routing, line_station)
            result_time_list.append(str(hour) + ':' + str(remain_min) + ':' + str(remain_second) + ';' + str(routing_num) + ';' + str(line_id) + ';' + str(timetable_routing_num))
    return result_time_list

# 给定线路编号，交路，判断当前区间是否在交路上
def fuc_is_in_routing(line_id: str, routing_num: int, section: tuple, routing: list, line_station: list):
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x!= None, line_station[i]))
    for i in routing:
        if line_id == i[0] and routing_num == int(i[1]):
            ostation, dstation = i[2].split(',')[0], i[3].split(',')[0]
    for i in line_station:
        if i[0] == line_id:
            temp_line_station = i[1:]
            a1 = temp_line_station.index(ostation)
            a2 = temp_line_station.index(dstation)
            if a1 < a2:
                temp_line_station1 = temp_line_station[a1: a2 + 1]
            else:
                temp_line_station1 = temp_line_station[a2: a1 + 1]
            if section[0] in temp_line_station1 and section[1] in temp_line_station1:
                return True
            else:
                return False

# 给定车辆到达时间列表，容忍时间，区间，得到列数
def get_train_num(timetable: list, work_rest: int, line_station: list, running_time: list, routing: list, now_time, tolerate_time, all_count: int):
    section_all, section_dict = fuc_section_gen(line_station)
    all_n, n = len(section_all), 0
    result_num = {}
    nhour, nmins, nsecond = int(now_time.split(':')[0]), int(now_time.split(':')[1]), int(now_time.split(':')[2])
    phour, pmins, psecond = int(tolerate_time.split(':')[0]), int(tolerate_time.split(':')[1]), int(tolerate_time.split(':')[2])
    n_s, p_s = nhour*60*60 + nmins*60 + nsecond, phour*60*60 + pmins*60 + psecond
    for i in section_all:
        print('当前总进度:' + str(all_count) + '/69  ' + str(all_count/69*100) + '%    ' + '当前车流进度：' + str(n/all_n*100) + '%')
        n += 1
        now_train_num = 0
        time_list = get_train_time(i, timetable, work_rest, line_station, running_time, routing)
        for j in time_list:
            hmsr = j.split(';')
            hms, routing_num, line_id, xxx = hmsr[0].split(':'), int(hmsr[1]), hmsr[2], int(hmsr[3])
            thour, tmins, tsecond = int(hms[0]), int(hms[1]), int(hms[2])
            if fuc_is_in_routing(line_id, routing_num, i, routing, line_station) is True and section_dict[i] == xxx:
                t_s = thour*60*60 + tmins*60 + tsecond
                if t_s >= n_s and t_s <= p_s:
                    now_train_num += 1
        result_num[i] = now_train_num
    return result_num

# 处理环线数据问题
def re_wirte_result(num_dict, line_station):
    section_all, section_dict = fuc_section_gen(line_station)
    s2n, s2s, s10n, s10s = num_dict[('西直门', '车公庄')], num_dict[('西直门', '积水潭')], num_dict[('巴沟', '火器营')], num_dict[('巴沟', '苏州街')]
    for i in section_all:
        line_id = fuc_identify_line_id(i, line_station)
        if line_id[0] == 'S020' and section_dict[i] == section_dict[('西直门', '车公庄')]:
            num_dict[i] = s2n
        elif line_id[0] == 'S020' and section_dict[i] == section_dict[('西直门', '积水潭')]:
            num_dict[i] = s2s
        elif line_id[0] == 'S100' and section_dict[i] == section_dict[('巴沟', '火器营')]:
            num_dict[i] = s10n
        elif line_id[0] == 'S100' and section_dict[i] == section_dict[('巴沟', '苏州街')]:
            num_dict[i] = s10s
    return num_dict

#-----------------------------------------------------------------------------------以上车流数据处理方法，以下客流数据处理方法

# 线路网络模型构建
def fuc_line_net_gen(line_net: list):
    line_networks = nx.Graph()
    line_nodes = ['S010', 'S020', 'S040', 'S050', 'S051', 'S060', 'S070', 'S080', 'S090', 'S091', 'S100', 'S110', 'S130', 'S131', 'S141', 'S150', 'S160', 'S161', 'S170']
    for i in line_nodes:
        line_networks.add_node(i)
    for i in line_net:
        line_networks.add_edge(i[0], i[1], weight = 1)
    return line_networks

# 线路网络模型最小换乘次数(Floyd)
def fuc_net_floyd(o_line_id: str, d_line_id: str, line_net: list):
    line_networks = fuc_line_net_gen(line_net)
    xxx = nx.all_shortest_paths(line_networks, source=o_line_id, target=d_line_id)
    return list(xxx)

# 给定区间，返回距离
def fuc_get_distance(o_station: str, d_station: str, distance: list):
    for i in distance:
        if o_station in i and d_station in i:
            return int(i[2])

# 车站网络模型构建
def fuc_node_net_gen(line_station: list, distance: list):
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x!= None, line_station[i]))
    node_networks = nx.Graph()
    for i in line_station:
        if i[0] not in ['S052', 'S092', 'S101', 'S102', 'S190', 'S021']:
            temp_list = i[1:]
            if i[0] not in ['S020', 'S100']:
                for j in range(len(temp_list)):
                    if j < len(temp_list) - 1:
                        node_networks.add_edge(temp_list[j], temp_list[j + 1], weight = fuc_get_distance(temp_list[j], temp_list[j + 1], distance))
            elif i[0] in ['S020', 'S100']:
                for j in range(len(temp_list)):
                    node_networks.add_edge(temp_list[j - 1], temp_list[j], weight = fuc_get_distance(temp_list[j - 1], temp_list[j], distance))
    return node_networks

# 给定两个线路ID，返回两线路共同站点
def fuc_same_station(o_line_id: str, d_line_id: str, line_station: list):
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x!= None, line_station[i]))
    same_station_list = []
    for i in line_station:
        if i[0] == o_line_id:
            temp_a = i[1:]
        elif i[0] == d_line_id:
            temp_b = i[1:]
    for i in temp_a:
        if i in temp_b:
            same_station_list.append(i)
    return same_station_list

# 给定某线路上的起点与终点，返回两点间所有区间与两点
def fuc_between_get(o_station: str, d_station: str, line_station: list, distance: list):
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x!= None, line_station[i]))
    result_section, result_length, the_line_id = [], 0, 0
    o_line_id, d_line_id = fuc_identify_line_id2(o_station, line_station), fuc_identify_line_id2(d_station, line_station)
    for i in o_line_id:
        if i in d_line_id:
            the_line_id = i
    for i in line_station:
        if i[0] == the_line_id:
            temp_list = i[1:]
            a1 = temp_list.index(o_station)
            a2 = temp_list.index(d_station)
            if a1 < a2: temp_list = temp_list[a1: a2 + 1]
            elif a1 > a2: temp_list = list(reversed(temp_list[a2: a1 + 1]))
            for j in range(len(temp_list)):
                if j < len(temp_list) - 1:
                    result_section.append((temp_list[j], temp_list[j + 1]))
    for i in result_section:
        for j in distance:
            if i[0] in j and i[1] in j:
                result_length += int(j[2])
    return result_section, result_length

# logit概率分配
def fuc_logit(section_list: list, length_list: list):
    result_list = []
    min_length = min(length_list)
    temp_section, temp_path = [], []
    for i in range(len(length_list)):
        if length_list[i] <= 1.5 * min_length:
            temp_section.append(section_list[i])
            temp_path.append(length_list[i])
    temp_path = np.array(temp_path)
    temp_path = np.exp(-0.002*temp_path)
    temp_path = temp_path/np.sum(temp_path)
    for i in range(len(temp_path)):
        result_list.append((temp_section[i], temp_path[i]))
    return result_list

# 最小换成次数的广度优先搜索最短路算法
def fuc_min_interchange_shortest_path(o_station: str, d_station: str, line_station: list, distance: list, line_net: list):
    result_list_with_p, result_section_list, re, record_trans_station = [], [], [], []
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x!= None, line_station[i]))
    o_line_id, d_line_id = fuc_identify_line_id2(o_station, line_station), fuc_identify_line_id2(d_station, line_station)
    for i in o_line_id:
        for j in d_line_id:
            min_interchange_list = fuc_net_floyd(i, j, line_net)
            for one_min_interchange_plan in min_interchange_list:
                record_trans_station = []
                # 无需换乘时
                if len(one_min_interchange_plan) == 1:
                    result_section_list, route_length = fuc_between_get(o_station, d_station, line_station, distance)
                    return [(result_section_list, 1)]
                # 需要换乘时
                else:
                    ori_symbol = ['x']
                    for k in range(len(one_min_interchange_plan)):
                        if k < len(one_min_interchange_plan) - 1:
                            trans_station_list = fuc_same_station(one_min_interchange_plan[k], one_min_interchange_plan[k + 1], line_station)
                            record_trans_station.append(trans_station_list)
                    # print(one_min_interchange_plan)
                    # print(record_trans_station)
                    for k in range(len(record_trans_station)):
                        ori_symbol = list(itertools.product(ori_symbol, record_trans_station[k]))
                    for p in ori_symbol:
                        re.append(str(p).replace('(', '').replace(')', '').replace('\'', '').replace(' ', '').split(','))
    for i in range(len(re)):
        re[i][0] = o_station
        re[i].append(d_station)
    record_path_length = []
    record_section = []
    for i in range(len(re)):
        path_length = 0
        section_list = []
        for j in range(len(re[i])):
            if j < len(re[i]) - 1:
                t_sec_l, t_path_l = fuc_between_get(re[i][j], re[i][j + 1], line_station, distance)
                path_length += t_path_l
                section_list += t_sec_l
        record_path_length.append(path_length)
        record_section.append(section_list)
    result_list = fuc_logit(record_section, record_path_length)
    return result_list

# 筛选并处理OD数据
def fuc_filter_OD(OD: list, now_time: str, tolerate_sec: int):
    hms = now_time.split(':')
    now_sec = int(hms[0])*60*60 + int(hms[1])*60 + int(hms[2])
    OD_time_list = OD[0][2:]
    for i in range(len(OD_time_list)): OD_time_list[i] = int(OD_time_list[i])
    column_list = []
    OD.pop(0)
    for i in OD_time_list:
        if i >= now_sec - tolerate_sec and i <= now_sec + tolerate_sec:
            column_list.append(OD_time_list.index(i) + 2)
    result_OD_list = []
    for i in OD:
        temp_list = []
        temp_person_sum = 0
        temp_list.append((i[0], i[1]))
        for j in column_list:
            temp_person_sum += int(i[j])
        temp_list.append(temp_person_sum)
        result_OD_list.append(temp_list)
    return result_OD_list

# 给定筛选OD客流，当前时间，返回区间乘客数
def get_passenger_num(now_time: str, tolerate_sec: int, OD: list, line_station: list, distance: list, line_net: list, all_count: int):
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x != None, line_station[i]))
    result_passenger_dict = {}
    section_all, section_dict = fuc_section_gen(line_station)
    for i in section_all:
        result_passenger_dict[i] = 0
    OD_list_filter = fuc_filter_OD(OD, now_time, tolerate_sec)
    count1 = 1
    for i in OD_list_filter: # 对于每一行od来说，都要计算一次logit，对这里代码需要增加内容
        print(i)
        input('T?')
        print('当前总进度:' + str(all_count) + '/69  ' + str(all_count/69*100) +  '%    ' + '当前客流进度:' + str(count1/112290*100) + '%')
        count1 += 1
        add_list = fuc_min_interchange_shortest_path(i[0][0], i[0][1], line_station, distance, line_net)
        for j in add_list:
            section_list, p = j[0], j[1]
            for section in section_list:
                result_passenger_dict[section] += i[1]*p
    return result_passenger_dict

# qij值计算
def couple_qij(train_num_dict: dict, passenger_num_dict: dict, line_station: list, train_capacity: int):
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x != None, line_station[i]))
    section_all, section_dict = fuc_section_gen(line_station)
    result_qij_dict = {}
    for i in section_all:
        if train_num_dict[i] != 0:
            result_qij_dict[i] = 1 - passenger_num_dict[i]/(train_num_dict[i]*train_capacity)
        else:
            result_qij_dict[i] = 0
    return result_qij_dict

#-----------------------------------------------------------------------------------以下多线程方法

# 生成多线程使用的列表
def fuc_time_list_gen(start_hour, end_hour):
    result_list1, result_list2 = [], []
    for i in range(end_hour + 1):
        if i >= start_hour:
            result_list1.append(str(i) + ':00:00')
            result_list1.append(str(i) + ':15:00')
            result_list1.append(str(i) + ':30:00')
            result_list1.append(str(i) + ':45:00')
            result_list2.append(str(i) + ':30:00')
            result_list2.append(str(i) + ':45:00')
            result_list2.append(str(i + 1) + ':00:00')
            result_list2.append(str(i + 1) + ':15:00')
    return result_list1, result_list2

# 多线程融合方法
def processA(items):
    now_time = items[0]
    train_tolerate_time = items[1]
    passenger_select_second = items[2]
    train_capacity = items[3]
    work_rest = items[4]
    timetable = items[5]
    line_station = items[6]
    running_time = items[7]
    routing = items[8]
    OD = items[9]
    station_distance = items[10]
    line_net_connection = items[11]
    train_num_dict = re_wirte_result(get_train_num(timetable, work_rest, line_station, running_time, routing, now_time, train_tolerate_time), line_station)
    passenger_num_dict = get_passenger_num(now_time, passenger_select_second, OD, line_station, station_distance, line_net_connection)
    q_ij = couple_qij(train_num_dict, passenger_num_dict, line_station, train_capacity)
    return q_ij

def fuc_multiprocess_list_gen(now_time_list: list, train_trolerate_time_list: list, passenger_select_second: int, train_capacity: int, work_rest: int, timetable, line_station, running_time, routing, OD, station_distance, line_net_connection):
    result_list = []
    l = len(now_time_list)
    for i in range(l):
        new_list = []
        new_list.append(now_time_list[i])
        new_list.append(train_trolerate_time_list[i])
        new_list.append(passenger_select_second)
        new_list.append(train_capacity)
        new_list.append(work_rest)
        new_list.append(timetable)
        new_list.append(line_station)
        new_list.append(running_time)
        new_list.append(routing)
        new_list.append(OD)
        new_list.append(station_distance)
        new_list.append(line_net_connection)
        result_list.append(new_list)
    return result_list

#-----------------------------------------------------------------------------------以下结果处理

# 节点位置数据处理
def fuc_nodes_pos(pos_list: list):
    result_dict = {}
    for i in pos_list:
        xy = i[1].split(',')
        result_dict[i[0]] = (float(xy[0]), float(xy[1]))
    return result_dict

# 结果网络构建，给定一个txt文件，输出网络链接
def fuc_re_net_gen(line_station: list, file_path: str):
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x!= None, line_station[i]))
    f = open(file_path, encoding='utf-8', mode='r')
    q_ij = eval(f.read())
    f.close()
    node_networks = nx.Graph()
    for i in line_station:
        if i[0] not in ['S052', 'S092', 'S101', 'S102', 'S190', 'S021']:
            temp_list = i[1:]
            if i[0] not in ['S020', 'S100']:
                for j in range(len(temp_list)):
                    if j < len(temp_list) - 1:
                        node_networks.add_edge(temp_list[j], temp_list[j + 1], weight = (q_ij[(temp_list[j], temp_list[j + 1])] + q_ij[(temp_list[j + 1], temp_list[j])])/2)
            elif i[0] in ['S020', 'S100']:
                for j in range(len(temp_list)):
                    node_networks.add_edge(temp_list[j - 1], temp_list[j], weight = (q_ij[(temp_list[j - 1], temp_list[j])] + q_ij[(temp_list[j], temp_list[j - 1])])/2)
    return node_networks

def fuc_re_net_gen2(line_station: list, file_path: str):
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x!= None, line_station[i]))
    f = open(file_path, encoding='utf-8', mode='r')
    q_ij = eval(f.read())
    f.close()
    node_networks = nx.Graph()
    for i in line_station:
        if i[0] not in ['S052', 'S092', 'S101', 'S102', 'S190', 'S021']:
            temp_list = i[1:]
            if i[0] not in ['S020', 'S100']:
                for j in range(len(temp_list)):
                    if j < len(temp_list) - 1:
                        node_networks.add_edge(temp_list[j], temp_list[j + 1], weight = q_ij[(temp_list[j], temp_list[j + 1])] + q_ij[(temp_list[j + 1], temp_list[j])])
                        # if q_ij[(temp_list[j], temp_list[j + 1])] <= q_ij[(temp_list[j + 1], temp_list[j])]:
                        #     node_networks.add_edge(temp_list[j], temp_list[j + 1], weight = q_ij[(temp_list[j], temp_list[j + 1])] - q_ij[(temp_list[j + 1], temp_list[j])])
                        # else:
                        #     node_networks.add_edge(temp_list[j], temp_list[j + 1], weight = q_ij[(temp_list[j + 1], temp_list[j])] - q_ij[(temp_list[j], temp_list[j + 1])])
            elif i[0] in ['S020', 'S100']:
                for j in range(len(temp_list)):
                    node_networks.add_edge(temp_list[j - 1], temp_list[j], weight = q_ij[(temp_list[j - 1], temp_list[j])] + q_ij[(temp_list[j], temp_list[j - 1])])
                    # if q_ij[(temp_list[j - 1], temp_list[j])] <= q_ij[(temp_list[j], temp_list[j - 1])]:
                    #     node_networks.add_edge(temp_list[j - 1], temp_list[j], weight = q_ij[(temp_list[j - 1], temp_list[j])] - q_ij[(temp_list[j], temp_list[j - 1])])
                    # else:
                    #     node_networks.add_edge(temp_list[j - 1], temp_list[j], weight = q_ij[(temp_list[j], temp_list[j - 1])] - q_ij[(temp_list[j - 1], temp_list[j])])
    return node_networks

# 聚类染色，输出颜色列表
def fuc_nodes_color(graph: nx.Graph):
    color_list = []
    node_list = list(nx.nodes(graph))
    for i in range(len(node_list)): color_list.append('#BABABA')
    cluster_dict = nx.connected_components(graph)
    sort_list = sorted(cluster_dict, key = len, reverse=True)
    if len(sort_list) == 1:
        for i in range(len(color_list)):
            color_list[i] = '#14D21F'
        return color_list
    elif len(sort_list) == 2:
        for i in node_list:
            for j in sort_list[0]:
                if i == j:
                    color_list[node_list.index(i)] = '#14D21F'
            for j in sort_list[1]:
                if i == j:
                    color_list[node_list.index(i)] = '#00A0FF'
        return color_list
    elif len(sort_list) == 3:
        for i in node_list:
            for j in sort_list[0]:
                if i == j:
                    color_list[node_list.index(i)] = '#14D21F'
            for j in sort_list[1]:
                if i == j:
                    color_list[node_list.index(i)] = '#00A0FF'
            for j in sort_list[2]:
                if i == j:
                    color_list[node_list.index(i)] = '#FF6C14'
        return color_list
    elif len(sort_list) >= 4:
        for i in node_list:
            for j in sort_list[0]:
                if i == j:
                    color_list[node_list.index(i)] = '#14D21F'
            for j in sort_list[1]:
                if i == j:
                    color_list[node_list.index(i)] = '#00A0FF'
            for j in sort_list[2]:
                if i == j:
                    color_list[node_list.index(i)] = '#FF6C14'
            for j in sort_list[3]:
                if i == j:
                    color_list[node_list.index(i)] = '#FF2B6C'
        return color_list
    
def fuc_delta_rij(file_path: str, line_station: list, num: int):
    net = fuc_re_net_gen2(line_station, file_path)
    net.remove_edge('鼓楼大街', '积水潭')
    net.remove_edge('十里河', '潘家园')
    w_list = nx.get_edge_attributes(net, 'weight')
    temp_list = sorted(zip(w_list.values(), w_list.keys()),reverse=False)
    # print(temp_list[0])
    result_list, result_num, count = [], [], 1
    for i in temp_list:
        if i[0] > 0 and i[0] < 1:
            result_num.append(i[0])
            result_list.append(i[1])
            if count > num - 1:
                break
            count += 1
        else: 
            continue
    return result_num, result_list

def fuc_metric_con(file_path: str, line_station: list):
    net = fuc_re_net_gen(line_station, file_path)
    # net = func_enhance_net_gen(line_station, file_path, 1.5)
    l = len(list(net.nodes))
    q, sq = 0, 0.01
    Gq_list, Gq_imp_list, Gq_ran_list = [], [], []
    SGq_list = []
    while q < 1:
        w_list = nx.get_edge_attributes(net, 'weight')
        for u, v in net.edges():
            if w_list[(u, v)] < q:
                net.remove_edge(u, v)
        SGq_list.append(nx.local_efficiency(net))
        largest_cc = len(max(nx.connected_components(net), key=len))
        Gq_list.append(largest_cc/nx.number_of_nodes(net))
        q += sq
    origin_re = sum(Gq_list)/len(Gq_list)
    SG_re = sum(SGq_list)/len(SGq_list)
    return origin_re, SG_re

def fuc_metric_rob(line_station: list, file_path: str, improve_value, pen_value, q):
    net = fuc_re_net_gen(line_station, file_path)
    random_net = fuc_re_net_gen(line_station, file_path)
    improve_net = fuc_re_net_gen(line_station, file_path)
    count = 1
    w_list = nx.get_edge_attributes(random_net, 'weight')
    for u, v in random_net.edges():
        if random.random() < 0.12:
            random_net.add_edge(u, v, weight = w_list[(u, v)] + improve_value)
            count += 1
        if count > 12:
            break
    w_list = nx.get_edge_attributes(improve_net, 'weight')
    for u, v in improve_net.edges():
        if (u, v) in [('金台路', '十里堡'), ('青年路', '褡裢坡'), ('十里堡', '青年路'), ('褡裢坡', '黄渠'), ('黄渠', '常营'), ('大望路', '四惠'), \
                      ('建国门', '永安里'), ('永安里', '国贸'), ('雍和宫', '北新桥'), ('北新桥', '张自忠路'), ('张自忠路', '东四'), ('菜市口', '陶然亭'), \
                        ('陶然亭', '北京南站'), ('军事博物馆', '木樨地'), ('木樨地', '南礼士路'), ('南礼士路', '复兴门'), ('北土城', '健德门'), ('健德门', '牡丹园'), \
                        ('牡丹园', '西土城'), ('西土城', '知春路'), ('清河', '上地'), ('上地', '五道口'), ('五道口', '知春路'), ('朱辛庄', '生命科学园'), \
                        ('生命科学园', '西二旗'), ('霍营', '育新'), ('育新', '西小口'), ('西小口', '永泰庄'), ('永泰庄', '林萃桥'), ('林萃桥', '森林公园南门'), \
                        ('森林公园南门', '奥林匹克公园')]:
            improve_net.add_edge(u, v, weight = w_list[(u, v)] + improve_value)
    for u, v in net.edges():
        t1 = nx.get_edge_attributes(net, 'weight')[(u, v)]
        t2 = nx.get_edge_attributes(improve_net, 'weight')[(u, v)]
        t3 = nx.get_edge_attributes(random_net, 'weight')[(u, v)]
        if t1 < q:
            net.remove_edge(u, v)
        if t2 < q:
            improve_net.remove_edge(u, v)
        if t3 < q:
            random_net.remove_edge(u, v)
    a, b, c = len(max(nx.connected_components(net), key=len)), len(max(nx.connected_components(improve_net), key=len)), len(max(nx.connected_components(random_net), key=len))
    for u, v in net.edges():
        t1 = nx.get_edge_attributes(net, 'weight')[(u, v)]
        t2 = nx.get_edge_attributes(improve_net, 'weight')[(u, v)]
        t3 = nx.get_edge_attributes(random_net, 'weight')[(u, v)]
        if t1 < q + pen_value:
            net.remove_edge(u, v)
        if t2 < q + pen_value:
            improve_net.remove_edge(u, v)
        if t3 < q + pen_value:
            random_net.remove_edge(u, v)
    a1, b1, c1 = len(max(nx.connected_components(net), key=len)), len(max(nx.connected_components(improve_net), key=len)), len(max(nx.connected_components(random_net), key=len))
    origin_rob, improve_rob, random_rob = 1-(a-a1)/nx.number_of_nodes(net), 1-(b - b1)/nx.number_of_nodes(improve_net), 1-(c - c1)/nx.number_of_nodes(random_net)
    return origin_rob, improve_rob, random_rob

def fuc_metric_vul(file_path: str, line_station: list, improve_value, q):
    net = fuc_re_net_gen(line_station, file_path)
    # net = func_enhance_net_gen(line_station, file_path, 1.5)
    l = len(list(net.nodes))
    temp_dict = nx.get_edge_attributes(net, 'weight')
    for u, v in net.edges():
        t1 = temp_dict[(u, v)]
        if t1 < q:
            net.remove_edge(u, v)
    a = len(list(nx.connected_components(net))) / l
    # a = nx.number_connected_components(net)/nx.number_of_nodes(net)
    return a

def fuc_metric_rec(file_path: str, line_station: list, improve_value, q, desire_persentage):
    net = fuc_re_net_gen(line_station, file_path)
    random_net = fuc_re_net_gen(line_station, file_path)
    improve_net = fuc_re_net_gen(line_station, file_path)
    count = 1
    w_list = nx.get_edge_attributes(random_net, 'weight')
    for u, v in random_net.edges():
        if random.random() < 0.12:
            random_net.add_edge(u, v, weight = w_list[(u, v)] + improve_value)
            count += 1
        if count > 12:
            break
    w_list = nx.get_edge_attributes(improve_net, 'weight')
    for u, v in improve_net.edges():
        if (u, v) in [('金台路', '十里堡'), ('青年路', '褡裢坡'), ('十里堡', '青年路'), ('褡裢坡', '黄渠'), ('黄渠', '常营'), ('大望路', '四惠'), \
                      ('建国门', '永安里'), ('永安里', '国贸'), ('雍和宫', '北新桥'), ('北新桥', '张自忠路'), ('张自忠路', '东四'), ('菜市口', '陶然亭'), \
                        ('陶然亭', '北京南站'), ('军事博物馆', '木樨地'), ('木樨地', '南礼士路'), ('南礼士路', '复兴门'), ('北土城', '健德门'), ('健德门', '牡丹园'), \
                        ('牡丹园', '西土城'), ('西土城', '知春路'), ('清河', '上地'), ('上地', '五道口'), ('五道口', '知春路'), ('朱辛庄', '生命科学园'), \
                        ('生命科学园', '西二旗'), ('霍营', '育新'), ('育新', '西小口'), ('西小口', '永泰庄'), ('永泰庄', '林萃桥'), ('林萃桥', '森林公园南门'), \
                        ('森林公园南门', '奥林匹克公园')]:
            improve_net.add_edge(u, v, weight = w_list[(u, v)] + improve_value)
    net_c = copy.deepcopy(net)
    random_net_c = copy.deepcopy(random_net)
    improve_net_c = copy.deepcopy(improve_net)
    for u, v in net.edges():
        t1 = nx.get_edge_attributes(net, 'weight')[(u, v)]
        t2 = nx.get_edge_attributes(improve_net, 'weight')[(u, v)]
        t3 = nx.get_edge_attributes(random_net, 'weight')[(u, v)]
        if t1 < q:
            net.remove_edge(u, v)
        if t2 < q:
            improve_net.remove_edge(u, v)
        if t3 < q:
            random_net.remove_edge(u, v)
    t, rec1, rec2, rec3 = 0, 0, 0, 0
    while t < desire_persentage:
        for u, v in net_c.edges():
            net_c.add_edge(u, v, weight = nx.get_edge_attributes(net_c, 'weight')[(u, v)] + 0.01)
        net = copy.deepcopy(net_c)
        rec1 += 0.01
        w_list_temp = nx.get_edge_attributes(net, 'weight')
        for u, v in net.edges():
            if w_list_temp[(u, v)] < q: net.remove_edge(u, v)
        t = len(max(nx.connected_components(net), key=len))/nx.number_of_nodes(net)
    t = 0
    while t < desire_persentage:
        for u, v in improve_net_c.edges():
            improve_net_c.add_edge(u, v, weight = nx.get_edge_attributes(improve_net_c, 'weight')[(u, v)] + 0.01)
        improve_net = copy.deepcopy(improve_net_c)
        rec2 += 0.01
        w_list_temp = nx.get_edge_attributes(improve_net, 'weight')
        for u, v in improve_net.edges():
            if w_list_temp[(u, v)] < q: improve_net.remove_edge(u, v)
        t = len(max(nx.connected_components(improve_net), key=len))/nx.number_of_nodes(improve_net)
    t = 0
    while t < desire_persentage:
        for u, v in random_net_c.edges():
            random_net_c.add_edge(u, v, weight = nx.get_edge_attributes(random_net_c, 'weight')[(u, v)] + 0.01)
        random_net = copy.deepcopy(random_net_c)
        rec3 += 0.01
        w_list_temp = nx.get_edge_attributes(random_net, 'weight')
        for u, v in random_net.edges():
            if w_list_temp[(u, v)] < q: random_net.remove_edge(u, v)
        t = len(max(nx.connected_components(random_net), key=len))/nx.number_of_nodes(random_net)
    return 1-rec1, 1-rec2, 1-rec3

def table_2(sta: list):
    result_dict = {}
    for i in sta:
        if i not in result_dict.keys():
            result_dict[i] = 1
        else:
            result_dict[i] = result_dict[i] + 1
    result_list = sorted(zip(result_dict.values(), result_dict.keys()), reverse=True)
    return result_list

def fig_1(y_ax: list):
    x_ax = list(range(80))
    x_tick = []
    # for i in range(20):
    #     x_tick.append(str(i + 4) + ':00:00')
    #     # x_tick.append(str(i + 4) + ':15:00')
    #     # x_tick.append(str(i + 4) + ':30:00')
    #     # x_tick.append(str(i + 4) + ':45:00')
    #     x_tick.append(None)
    #     x_tick.append(None)
    #     x_tick.append(None)
    plt.style.use('classic')
    y_mean = np.array(y_ax).mean()
    x_tick.append('4:00')
    for i in range(78):
        x_tick.append(None)
    x_tick.append('23:45')
    plt.rcParams['font.sans-serif'] = 'Arial'
    axes = plt.axes([0, 0, 1.2, 0.3])
    axes.bar(x_ax, y_ax,align='center', color = '#628AE9', tick_label = x_tick, edgecolor = 'None', alpha = 0.8, zorder = 4)
    axes.set_xlim((0, 80))
    axes.set_ylim((0, 300000))
    # plt.grid(True, axis='y', ls = ':', color = 'black', alpha = 0.4, zorder = 1)
    # plt.axvspan(xmin=11, xmax=22, facecolor = '#6FFACC', alpha = 1, zorder = 3, edgecolor = 'None')
    # plt.axvspan(xmin=52, xmax=61, facecolor = '#6FFACC', alpha = 1, zorder = 3, edgecolor = 'None')
    plt.axhline(y=y_mean, color = '#A78B03', ls = '--', alpha = 0.8, label = 'Hourly inbound volume', zorder = 2, linewidth=4)
    plt.tick_params(axis='both', labelsize = 18, size = 0, width = 0, rotation = 45)
    # plt.annotate('Morning rush hour', xy = (16, 290000), xytext=(25, 275000), arrowprops={'arrowstyle': '->', 'connectionstyle': 'arc3', 'color': 'black'}, fontsize = 12)
    # plt.annotate('Evening rush hour', xy = (56, 290000), xytext=(25, 225000), arrowprops={'arrowstyle': '->', 'connectionstyle': 'arc3', 'color': 'black'}, fontsize = 12)
    # plt.annotate('Average hourly \n inbound volume', xy = (40, y_mean), xytext=(25, 110000), arrowprops={'arrowstyle': '->', 'connectionstyle': 'arc3', 'color': 'black'}, fontsize = 12)
    plt.savefig('D:/桌面/F/2022.12.05-北京地铁网络渗流研究/result_fig/fig.1/Fig.1.svg', transparent = True, bbox_inches = 'tight', pad_inches = 0.1)
    plt.clf()
    print('Fig.1 done!')

def fig_2_2(pos_list: list, line_station: list):
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x != None, line_station[i]))
    net = fuc_re_net_gen(line_station, 'D:/CodeSub/PYProgram/Beijing_Multi_Percolation/1.1result/6_00_00.txt')
    color_list = []
    node_list = nx.nodes(net)
    for i in node_list:
        # line_ids = fuc_identify_line_id2(i, line_station)
        # if len(line_ids) == 1:
        #     line_id = line_ids[0]
        #     if line_id == 'S010': color_list.append('#C23A30')
        #     elif line_id == 'S020': color_list.append('#006098')
        #     elif line_id == 'S040': color_list.append('#008E9C')
        #     elif line_id == 'S050': color_list.append('#A6217F')
        #     elif line_id == 'S051': color_list.append('#E40077')
        #     elif line_id == 'S060': color_list.append('#D29700')
        #     elif line_id == 'S070': color_list.append('#F6C582')
        #     elif line_id == 'S080': color_list.append('#009B6B')
        #     elif line_id == 'S090': color_list.append('#8FC31F')
        #     elif line_id == 'S091': color_list.append('#E46022')
        #     elif line_id == 'S100': color_list.append('#009BC0')
        #     elif line_id == 'S110': color_list.append('#ED796B')
        #     elif line_id == 'S130': color_list.append('#F9E700')
        #     elif line_id == 'S131': color_list.append('#DE82B2')
        #     elif line_id == 'S141': color_list.append('#D5A7A1')
        #     elif line_id == 'S150': color_list.append('#5B2C68')
        #     elif line_id == 'S160': color_list.append('#76A32E')
        #     elif line_id == 'S161': color_list.append('#B35A20')
        #     elif line_id == 'S170': color_list.append('#00A9A9')
        #     else: color_list.append('#000000')
        # else:
        #     color_list.append('#000000')
        color_list.append('#000000')
    nx.draw_networkx(net, pos = fuc_nodes_pos(pos_list), node_size = 5, with_labels = False, node_color = color_list, width = 0.5)
    plt.savefig('D:/桌面/r.jpg', dpi = 2400, bbox_inches = 'tight', pad_inches = 0)
    print('Fig.2.2 done!')

def fig_4_1(pos_list: list, line_station: list, file_path: str, save_path: str):
    q, q_step = 0.00, 0.01
    count_ = 1
    node_pos_dict = fuc_nodes_pos(pos_list)
    # plt.style.use('classic')
    while q < 1:
        net = fuc_re_net_gen(line_station, file_path)
        w_list = nx.get_edge_attributes(net, 'weight')
        for u, v in net.edges():
            w_now = w_list[(u, v)]
            if w_now < q:
                net.remove_edge(u, v)
        node_color_list = fuc_nodes_color(net)
        nx.draw_networkx_nodes(net, pos=node_pos_dict, node_size = 5, node_color = node_color_list)
        plt.axis('off')
        plt.savefig(save_path + '/N' + str(count_) + '.svg', transparent = True, bbox_inches = 'tight', pad_inches = 0.05)
        plt.clf()
        q += q_step
        count_ += 1
    print('Fig.4.1 done')

def fig_4_2(line_station: list, file_path: str, count_num: int):
    q_l, g_l, sg_l = [], [], []
    q, sq = 0.00, 0.01
    net = fuc_re_net_gen(line_station, file_path)
    n = nx.number_of_nodes(net)
    w_list = nx.get_edge_attributes(net, 'weight')
    while q < 1:
        for u, v in net.edges():
            w_now = w_list[(u, v)]
            if w_now < q:
                net.remove_edge(u, v)
        component_list = []
        for i in nx.connected_components(net):
            component_list.append(len(i))
        component_list.sort(reverse=True)
        if len(component_list) >= 2:
            g_l.append(component_list[0]/n)
            sg_l.append(component_list[1]/n)
        else:
            g_l.append(component_list[0]/n)
            sg_l.append(0)
        q_l.append(q)
        q += sq
    plt.rcParams['font.sans-serif'] = 'Arial'
    fig, ax1 = plt.subplots()
    line1, = ax1.plot(q_l, g_l, color = '#008AFF', marker = 'o', markersize = 8, label = '$G$', markerfacecolor = 'None')
    ax1.set_xlabel('$q$', fontsize = 20)
    ax1.set_ylabel('$G$', fontsize = 20)
    ax1.tick_params(axis='both', labelsize = 20, size = 0, width = 0)
    ax2 = ax1.twinx()
    line2, = ax2.plot(q_l, sg_l, color = '#E84444', marker = '^', markersize = 8, label = '$SG$', markerfacecolor = 'None')
    ax2.set_ylabel('$SG$', fontsize = 20)
    ax2.tick_params(axis='y', labelsize = 20, size = 0, width = 0)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.legend([line1, line2], ['$G$', '$SG$'], loc = 'upper right', fontsize = 15, frameon = False)
    plt.savefig('D:/桌面/F/2022.12.05-北京地铁网络渗流研究/result_fig/fig.4.2/' + str(count_num) + '.svg', transparent = True, bbox_inches = 'tight', pad_inches = 0.1)
    plt.clf()
    print('Fig.4.2 done')


def fig_4_3(fig43_data: list):
    y, x = fig43_data[0], fig43_data[1]
    x_tick = ['6:00']
    mean_qc = 0.7125
    for i in range(67): x_tick.append(None)
    x_tick.append('23:00')
    plt.figure(figsize=(15, 4))
    plt.xticks(x, x_tick)
    plt.ylim((0.01, 1))
    plt.xlim((0, 70))
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.tick_params(axis='both', labelsize = 12, size = 0, width = 0)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.axhline(y = mean_qc, color='#E20C12', ls = '--', alpha = 0.7, label = 'Average $q_c$')
    plt.plot(x, y, marker = 'H', color = '#009A30', markersize = '8', label = '$q_c$', markerfacecolor = 'None')
    plt.grid(True, axis='y', ls = ':', color = '#CF648E', alpha = 0.8)
    # plt.annotate('0.709', xy = (60, 0.7098), xytext=(59, 0.5), arrowprops={'arrowstyle': '->', 'connectionstyle': 'arc3', 'color': 'black'}, fontsize = 12)
    # plt.axhspan(ymin=0.647, ymax= 0.789, facecolor = '#038CA8', alpha = 0.2)
    plt.legend(fontsize = 12, frameon = False, loc = 'best')
    plt.savefig('D:/桌面/F/2022.12.05-北京地铁网络渗流研究/result_fig/fig.4.3/Fig.4.3.svg', transparent = True, bbox_inches = 'tight', pad_inches = 0.1)
    print('Fig.4.3 done')


def fig_4_4(fig44_data: list):
    x, y =list(range(69)), fig44_data
    x_tick = ['6:00:00']
    for i in range(67): x_tick.append(None)
    x_tick.append('23:00:00')
    plt.xlim((-2, 70))
    plt.xticks(x, x_tick)
    plt.rcParams['font.sans-serif'] = 'Palatino Linotype'
    plt.tick_params(axis='both', labelsize = 12, size = 0, width = 0)
    plt.plot(x, y, marker = 'o', markersize = 4, color = '#AD612A', label = '$\max\{|\Delta r_{ij}|\}$')
    plt.grid(True, axis='y', ls = ':', color = '#ECC69F', alpha = 1)
    plt.legend(fontsize = 12, frameon = False)
    plt.axvspan(xmin = 4, xmax = 15, facecolor = '#038CA8', alpha = 0.2)
    plt.savefig('D:/桌面/F/2022.12.05-北京地铁网络渗流研究/result_fig/fig.4.4/Fig.4.4.jpg', dpi = 2400, bbox_inches = 'tight', pad_inches = 0)
    print('Fig.4.4 done')

def func_line_fit(size_list: np.ndarray, p_list: np.ndarray):
    lg_x, lg_y = np.log10(size_list), np.log10(p_list)
    # lg_x, lg_y = list(lg_x), list(lg_y)
    # for i in range(len(lg_x)):
    #     if (lg_x[i] in [np.nan]) or (lg_y[i] in [np.nan]):
    #         lg_x.pop(i)
    #         lg_y.pop(i)
    a, b = np.polyfit(lg_x, lg_y, 1)
    print(a)
    result_x, result_y = [], []
    fit_list = np.linspace(0, max(lg_x), 10)
    for i in fit_list:
        result_x.append(10**i)
        result_y.append(10**(a*i + b))
    return result_x, result_y

def fig_4_5(x, y, line_x, line_y):
    # l = len(x_tick)
    # x = list(range(l))
    # x = x[:20]
    # y = y[:20]
    # plt.xticks(x, x_tick)
    plt.rcParams['font.sans-serif'] = 'Arial'
    axes = plt.axes([0, 0, 0.6, 0.35])
    axes.set_xlabel('Frequency of occurrence', fontsize = 18)
    axes.set_ylabel('Probability', fontsize = 18)
    axes.set_xscale('log')
    axes.set_yscale('log')
    plt.tick_params(axis='both', labelsize = 18, size = 0, width = 0)
    axes.scatter(x, y, edgecolors='#4E2A82', marker='^', color = 'white', s = 90, linewidths = 2.5, alpha=0.8)
    axes.plot(line_x, line_y, color = '#2A97D5', linewidth = 4, alpha = 0.8, linestyle = '-')
    # plt.bar(x, y, align='center', color = '#5394A8')
    # plt.grid(True, axis='y', ls = ':', color = '#E18F2B', alpha = 0.6)
    plt.savefig('D:/桌面/F/2022.12.05-北京地铁网络渗流研究/result_fig/fig.4.5/Fig.4.5.svg', transparent = True, bbox_inches = 'tight', pad_inches = 0.1)
    print('Fig.4.5 done')

def fig_4_10(origin_result: list):
    x = list(range(len(origin_result)))
    x_tick = ['6:00']
    for i in range(67): x_tick.append(None)
    x_tick.append('23:00')
    plt.xticks(x, x_tick)
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.tick_params(axis='both', labelsize = 12, size = 0, width = 0)
    # plt.xlim((-2, 15))
    # plt.ylim((0.5, 0.65))
    plt.plot(x, origin_result, marker = 'o', markersize = 8, markerfacecolor = 'white', color = '#00CAF9', label = 'Connectivity', linewidth = 2) # #00CAF9 #00A9FC
    # plt.plot(x, improve_result, marker = '^', markersize = 4, color = '#245B46', label = 'Improvement')
    # plt.plot(x, random_result, marker = '+', markersize = 10, color = '#483E11', alpha = 0.9, label = 'Random')
    # plt.grid(True, axis='y', ls = ':', color = '#483E11', alpha = 0.6)
    plt.legend(fontsize = 12, frameon = False)
    plt.savefig('D:/桌面/F/2022.12.05-北京地铁网络渗流研究/result_fig/fig.10/Fig.10.jpg', dpi = 2400, bbox_inches = 'tight', pad_inches = 0)
    print('Fig.10 done')

def fig_4_11(origin_result: list, improve_result: list, random_result: list, file_name):
    x = list(range(len(origin_result)))
    x_tick = ['6:00:00']
    for i in range(67): x_tick.append(None)
    x_tick.append('23:00:00')
    plt.xticks(x, x_tick)
    plt.rcParams['font.sans-serif'] = 'Palatino Linotype'
    # plt.xlim((40, 57))
    # plt.ylim((0, 0.3))
    plt.tick_params(axis='both', labelsize = 12, size = 0, width = 0)
    plt.plot(x, origin_result, marker = 'o', markersize = 4, color = '#F5B802', label = 'Origin')
    plt.plot(x, improve_result, marker = '^', markersize = 4, color = '#245B46', label = 'Improvement')
    # plt.plot(x, random_result, marker = '+', markersize = 4, color = '#483E11', alpha = 0.9, label = 'Random')
    plt.grid(True, axis='y', ls = ':', color = '#483E11', alpha = 0.6)
    plt.legend(fontsize = 12, frameon = False)
    plt.savefig('D:/桌面/F/2022.12.05-北京地铁网络渗流研究/result_fig/fig.11/q' + str(file_name) + '.jpg', dpi = 2400, bbox_inches = 'tight', pad_inches = 0)
    plt.clf()
    print('Fig.11 done')

def fig_4_13(origin_result: list):
    x = list(range(len(origin_result)))
    x_tick = ['6:00']
    for i in range(67): x_tick.append(None)
    x_tick.append('23:00')
    plt.xticks(x, x_tick)
    plt.rcParams['font.sans-serif'] = 'Arial'
    # plt.xlim((20, 39))
    # plt.ylim((0.28, 0.58))
    plt.ylim((-0.05, 1))
    plt.tick_params(axis='both', labelsize = 12, size = 0, width = 0)
    plt.plot(x, origin_result, marker = 'o', markersize = 8, markerfacecolor = 'white', color = '#7980E2', label = 'Fragmentation', linewidth = 2)
    # plt.plot(x, improve_result, marker = '^', markersize = 4, color = '#245B46', label = 'Improvement')
    # plt.plot(x, random_result, marker = '+', markersize = 4, color = '#483E11', alpha = 0.9, label = 'Random')
    # plt.grid(True, axis='y', ls = ':', color = '#483E11', alpha = 0.6)
    plt.legend(fontsize = 12, frameon = False, loc = 'upper left')
    plt.savefig('D:/桌面/F/2022.12.05-北京地铁网络渗流研究/result_fig/fig.13/Fig.13.jpg', dpi = 2400, bbox_inches = 'tight', pad_inches = 0)
    print('Fig.13 done')

def fig_4_14(origin_result: list, improve_result: list, random_result: list):
    x = list(range(len(origin_result)))
    x_tick = ['6:00']
    for i in range(67): x_tick.append(None)
    x_tick.append('23:00')
    plt.xticks(x, x_tick)
    plt.rcParams['font.sans-serif'] = 'Palatino Linotype'
    # plt.xlim((40, 57))
    plt.ylim((-0.05, 1.05))
    plt.tick_params(axis='both', labelsize = 20, size = 0, width = 0)
    plt.plot(x, origin_result, marker = 'o', markersize = 4, color = '#F5B802', label = 'Origin')
    plt.plot(x, improve_result, marker = '^', markersize = 4, color = '#245B46', label = 'Improvement')
    # plt.plot(x, random_result, marker = '+', markersize = 4, color = '#483E11', alpha = 0.9, label = 'Random')
    plt.grid(True, axis='y', ls = ':', color = '#483E11', alpha = 0.6)
    plt.legend(fontsize = 20, frameon = False, ncol = 1, loc = 'center right')
    plt.savefig('D:/桌面/F/2022.12.05-北京地铁网络渗流研究/result_fig/fig.14/Fig.14.jpg', dpi = 2400, bbox_inches = 'tight', pad_inches = 0)
    print('Fig.14 done')

def section_flow_num(origin_all_OD: list, line_station: list, distance: list, line_net: list):
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x!= None, line_station[i]))
    result_passenger_dict = {}
    section_all, section_dict = fuc_section_gen(line_station)
    for i in section_all:
        result_passenger_dict[i] = 0
    OD = []
    for i in origin_all_OD:
        temp_list = []
        temp_list.append((i[0], i[1]))
        temp_list.append(i[2])
        OD.append(temp_list)
    count1 = 1
    for i in OD:
        print('当前客流进度:' + str(count1/112290*100) + '%')
        count1 += 1
        add_list = fuc_min_interchange_shortest_path(i[0][0], i[0][1], line_station, distance, line_net)
        for j in add_list:
            section_list, p = j[0], j[1]
            for section in section_list:
                result_passenger_dict[section] += i[1]*p
    return result_passenger_dict


