import xlwings as xw
import numpy as np

origin_file_path = 'D:/桌面/F/2022.12.05-北京地铁网络渗流研究/all_data.xlsx'
new_app = xw.App(visible=True, add_book=False)
work_excel = new_app.books.open(origin_file_path)
origin_timetable = work_excel.sheets['timetable_data'].range('A2').expand('table').value
origin_routing = work_excel.sheets['routing_data'].range('A2').expand('table').value
origin_line_station = work_excel.sheets['line_station'].range('A1').expand('table').value
origin_running_time = work_excel.sheets['running_time'].range('A2').expand('table').value
origin_OD = work_excel.sheets['OD'].range('A1').expand('table').value

# ----------------------------------------
# 运行图数据正确性校验
def a(timetable: list):
    line_id, station_name, hour, mins = [], [], [], []
    for i in timetable:
        line_id.append(i[0])
        station_name.append(i[2])
        hour.append(i[3])
        mins.append(i[4])

    line_id = np.unique(np.array(line_id))
    if len(line_id) == 20:
        print('运行图线路ID数量校验通过')
    else:
        print('运行图线路ID数量出错')
        print(line_id)
    
    count1 = 0
    for i in range(len(hour)):
        try:
            hour[i] = int(hour[i])
        except:
            count1 += 1
            print('时间格式转换错误，数据位置：' + str(i))
    print('时间格式转换校验完成，共出现' + str(count1) + '处错误')

    count2 = 0
    for i in range(len(mins)):
        if type(mins[i]) is float or int:
            count2 += 0
        if type(mins[i]) is str:
            t = mins[i].split(',')
            for j in t:
                if j == None:
                    count2 += 1
                    print('分钟数据存在空值，数据位置：' + str(i))
                else:
                    jj = j.split(';')
                    if len(jj) > 2:
                        count2 += 1
                        print('分钟交路存在错误，数据位置：' + str(i))
                    elif len(jj[0]) != 2:
                        count2 += 1
                        print('分钟字段长度存在错误，数据位置：' + str(i))
                    elif len(jj) == 2 and len(jj[1]) > 2:
                        count2 += 1
                        print('分钟交路字段长度存在问题，数据位置：' + str(i))
                    else:
                        count2 += 0
    print('分钟与交路字段长度校验完成，共出现' + str(count2) + '处错误')

    count3 = 0
    for i in range(len(mins)):
        if type(mins[i]) is str:
            t = mins[i].split(',')
            empty_t = []
            for j in t:
                jx = int(j.split(';')[0])
                empty_t.append(jx)
            aa = empty_t.copy()
            aa.sort()
            if empty_t != aa:
                count3 += 1
                print('分钟数据未单调递增，数据位置：' + str(i))
    print('分钟数据单调性校验完成，共出现' + str(count3) + '处错误')

    if count1 + count2 + count3 == 0:
        print('运行图数据校验通过！')
    else:
        print('运行图数据校验失败，请重新检查')
# ----------------------------------------

# ----------------------------------------
# 运行图与交路数据对应性校验
def b(timetable: list, routing: list):
    count1 = 0
    test_routing_num = []
    routing_dict = {'S090': [0], 'S161': [0], 'S021': [0], 'S110': [0], 'S050': [0, 1], 'S070': [0, 1, 2, 3], 'S080': [0, 1, 2], 'S051': 
    [0, 1, 2], 'S130': [0, 1, 2, 3, 4], 'S150': [0, 1, 2, 3, 4, 5, 6], 'S131': [0, 1, 2, 3, 4], 'S091': [0, 1, 2], 'S060': [0, 1, 2, 3, 
    4, 5, 6, 7, 8, 9], 'S010': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], 'S040': [0, 1, 2, 3, 4], 'S141': [0, 1, 2], 'S160': [0], 
    'S170': [0], 'S020': [0, 1], 'S100': [0, 1]}
    for i in range(len(timetable)):
        if type(timetable[i][4]) is str:
            t1, t2 = timetable[i][0], timetable[i][4]
            min_list = t2.split(',')
            for j in min_list:
                jj = j.split(';')
                if len(jj) == 2:
                    test_routing_num.append([t1, jj[1], i])
    for i in test_routing_num:
        line_routing_list = routing_dict[i[0]]
        i[1] = int(i[1])
        if i[1] not in line_routing_list:
            count1 += 1
            print('运行图数据与交路不匹配，数据位置：' + str(i[2]))
    print('运行图与交路数据校验完成，共出现' + str(count1) + '处错误！')
    if count1 == 0:
        print('运行图与交路数据校验通过！')
    else:
        print('运行图与交路数据校验失败，请重新检查')
# ----------------------------------------

# ----------------------------------------
# 线路数据与运行时间对应性校验
def c(linestation, runningtime):
    linestation[13] = ['S100','潘家园','劲松','双井','国贸','金台夕照','呼家楼','团结湖','农业展览馆','亮马桥','三元桥','太阳宫','芍药居',
     '惠新西街南口','安贞门','北土城','健德门','牡丹园','西土城','知春路','知春里','海淀黄庄','苏州街','巴沟','火器营','长春桥','车道沟','慈寿寺',
     '西钓鱼台','公主坟','莲花桥','六里桥','西局','泥洼','丰台站','首经贸','纪家庙','草桥','角门西','角门东','大红门','石榴庄','宋家庄','成寿寺',
     '分钟寺','十里河']
    for i in range(len(linestation)):
        linestation[i] = list(filter(lambda x: x != None, linestation[i]))
    od_list = []
    for i in linestation:
        if i[0] not in ['S020', 'S021', 'S100', 'S052', 'S092', 'S101', 'S102', 'S190']:
            for j in range(len(i) - 1):
                if j < len(i) - 2:
                    od_list.append([i[j + 1], i[j + 2]])
                    od_list.append([i[j + 2], i[j + 1]])
        elif i[0] in ['S020', 'S100']:
            t_list = i[1:]
            for j in range(len(t_list)):
                od_list.append([t_list[j - 1], t_list[j]])
                od_list.append([t_list[j], t_list[j - 1]])
        elif i[0] == 'S021':
            t_list = [['北新桥', '东直门'], ['东直门', '北新桥'], ['东直门', '三元桥'], ['三元桥', '东直门'], ['三元桥', 'T3航站楼'], ['T3航站楼'
                    , '三元桥'], ['T3航站楼', 'T2航站楼'], ['T2航站楼', 'T3航站楼'], ['T2航站楼', '三元桥'], ['三元桥', 'T2航站楼']]
            for j in t_list:
                od_list.append(j)
    for i in range(len(runningtime)):
        runningtime[i] = runningtime[i][0:2]

    count3 = 0
    for i in range(int(len(od_list)/2)):
        if od_list[2*i] not in runningtime and od_list[2*i + 1] not in runningtime:
            count3 += 1
            print('存在未录入运营时间区间：' + str(od_list[2*i]) + str(od_list[2*i + 1]))
    for i in range(len(runningtime)):
        if runningtime[i] not in od_list:
            count3 += 1
            print('存在多余运营时间数据：' + str(i))
    print('线路与运行时间数据校验完成，共出现' + str(count3) + '处错误！')
    if count3 == 0:
        print('线路与运行时间数据对应性校验通过！')
    else:
        print('线路与运行时间数据校验失败！')
# ----------------------------------------

# ----------------------------------------
# OD车站名称校验
def d(line_station: list, OD: list):
    line_station[13] = ['S100','潘家园','劲松','双井','国贸','金台夕照','呼家楼','团结湖','农业展览馆','亮马桥','三元桥','太阳宫','芍药居',
     '惠新西街南口','安贞门','北土城','健德门','牡丹园','西土城','知春路','知春里','海淀黄庄','苏州街','巴沟','火器营','长春桥','车道沟','慈寿寺',
     '西钓鱼台','公主坟','莲花桥','六里桥','西局','泥洼','丰台站','首经贸','纪家庙','草桥','角门西','角门东','大红门','石榴庄','宋家庄','成寿寺',
     '分钟寺','十里河']
    for i in range(len(line_station)):
        line_station[i] = list(filter(lambda x: x != None, line_station[i]))
    need_list = []
    name_list = []
    count1 = 0
    for i in OD:
        name_list.append(i[0])
        name_list.append(i[1])
    for i in line_station:
        if i[0] not in ['S021', 'S052', 'S092', 'S101', 'S102', 'S190']:
            temp_list = i[1:]
            for j in temp_list:
                need_list.append(j)
    for i in range(len(name_list)):
        if name_list[i] not in need_list:
            print('第' + str(i/2) + '行站名不存在')
            count1 += 1
    print(count1)
    for i in range(len(need_list)):
        if need_list[i] not in name_list:
            print('存在缺失车站：' + str(need_list[i]))
# ----------------------------------------

# ----------------------------------------
# 距离数据校验
def e():
    return 0
# ----------------------------------------

# 运行图数据校验
# a(origin_timetable)
# b(origin_timetable, origin_routing)
# c(origin_line_station, origin_running_time)

# OD数据校验
# d(origin_line_station, origin_OD)

# 距离数据校验
