# collect the common function
# import script

def get_traffic_volume(traffic_file):
    # only support "cross" and "synthetic"
    if "cross" in traffic_file:
        sta = traffic_file.find("equal_") + len("equal_")
        end = traffic_file.find(".xml")
        return int(traffic_file[sta:end])
    elif "synthetic" in traffic_file:
        traffic_file_list = traffic_file.split("-")
        volume_list = []
        for i in range(2, 6):
            volume_list.append(int(traffic_file_list[i][2:]))

        vol = min(max(volume_list[0:2]), max(volume_list[2:]))

        return int(vol/100)*100
    elif "anon" in traffic_file:
        return int(traffic_file.split('_')[3].split('.')[0])