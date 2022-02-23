import numpy as np
import csv
import ast

def extract_gt_data(formation_number):
    filename = "datasets/2022_02_21/raw/Formation"+str(formation_number)+".csv"
    my_data = np.genfromtxt(filename, delimiter=',', skip_header=7)

    r_1 = my_data[:,2:5]
    r_2 = my_data[:,5:8]
    r_3 = my_data[:,8:11]

    return [r_1, r_2, r_3]

def extract_ts_data(formation_number,agent_number):
    prefix = "datasets/2022_02_21/raw/Formation"
    filename = prefix+str(formation_number)+"_Agent"\
               + str(agent_number)+"_timestamps.txt"
    
    mult_twr = -1

    ts_data = {}

    # Using readlines()
    file1 = open(filename, 'r')
    Lines = file1.readlines()

    for line in Lines:
        if line[0] == "A":
            target_agent = line[6]
            mult_twr = int(line[-2])
            if target_agent not in ts_data:
                ts_data[target_agent] = [np.empty((0,7)),
                                         np.empty((0,7)),
                                         np.empty((0,7))]
        elif "range" in line:
            row = ast.literal_eval(line)
            neighbour = str(row["neighbour"])
            if mult_twr == 0:
                temp = np.array([row["range"],
                                 row["tx1"],row["rx1"],
                                 row["tx2"],row["rx2"],
                                 0,0])
            else:
                temp = np.array([row["range"],
                                 row["tx1"],row["rx1"],
                                 row["tx2"],row["rx2"],
                                 row["tx3"],row["rx3"]])
            ts_data[neighbour][mult_twr]\
                = np.vstack((ts_data[neighbour][mult_twr], temp))

    return ts_data

def calculate_mean_gt_distance(r):
    d_12 = np.linalg.norm(r[0] - r[1],axis=1)
    d_12 = d_12[~np.isnan(d_12)]
    d_12 = np.mean(d_12)

    d_13 = np.linalg.norm(r[0] - r[2],axis=1)
    d_13 = d_13[~np.isnan(d_13)]
    d_13 = np.mean(d_13)

    d_23 = np.linalg.norm(r[1] - r[2],axis=1)
    d_23 = d_23[~np.isnan(d_23)]
    d_23 = np.mean(d_23)

    return [d_12, d_13, d_23]

def calculate_mean_range(formation,range1,range2,mult_twr):
    r12 = range1["2"][mult_twr][:,0]
    r12 = r12[~np.isnan(r12)]
    r12 = np.mean(r12)

    r13 = range1["3"][mult_twr][:,0]
    r13 = r13[~np.isnan(r13)]
    r13 = np.mean(r13)

    r23 = range2["3"][mult_twr][:,0]
    r23 = r23[~np.isnan(r23)]
    r23 = np.mean(r23)

    return np.array([r12,r13,r23])

def match_gt_with_range(gt_distance,range):
    '''
    We know in this experiment only tags 2 and 3 are interchangable,
    So r23 is in the right spot, need to find which is which between 
    r12 and r13.
    '''
    e1 = np.linalg.norm(gt_distance[0:2]-range[0:2])
    e2 = np.linalg.norm(gt_distance[0:2]-[range[1],range[0]])
    if e1-e2>0.05:
        return [gt_distance[1],gt_distance[0],gt_distance[2]]
    else:
        return gt_distance

def process_multitag_data(gt, ts_data, neighbours, agent, mult_twr): 
    empty = np.nan;
    col0a = np.array([agent])
    col0b = np.array([empty])
    
    # TODO: generalize this
    if agent == 2:
        agent_idx = 2 # fix this
    else:
        agent_idx = 0


    data = np.hstack(((col0a,col0b)))
    data = np.reshape(data,(1,2))

    for lv0 in range(len(neighbours)):
        col1 = np.array([neighbours[lv0]])
        col2 = np.empty((0,1))
        col3 = np.empty((0,1))
        col4_to_9 = np.empty((0,6))

        for formation in range(10):
            gt_formation = gt[formation,:]
            ts_formation = ts_data[formation][str(agent)][neighbours[lv0]][mult_twr]
            col4_to_9 = np.vstack((col4_to_9,ts_formation[:,1:]))
            n = np.size(ts_formation,0)
            
            col2_formation = np.reshape(np.array([10e10*formation]*n),(n,1))
            col2 = np.vstack((col2,col2_formation))

            col3_formation = np.reshape([gt_formation[lv0+agent_idx]]*n,(n,1))
            col3 = np.vstack((col3,col3_formation))

        n = np.size(col3,0)
        
        none_vector = np.reshape(np.array([empty]*(n-1)),(n-1,1))
        col1 = np.vstack((col1,none_vector))

        col10_to_11 = np.array([empty,empty]*n)
        col10_to_11 = np.reshape(col10_to_11, (n,2))

        data_new = np.hstack((col1, col2, col3, col4_to_9, col10_to_11))

        m = np.size(data,0)
        if np.size(data)>np.size(data,0):
            col_num = np.size(data,1)
        else:
            col_num = 1
        if m<n:
            none_matrix = np.array([empty]*col_num*(n-m))
            none_matrix = np.reshape(none_matrix, (n-m,col_num))
            data = np.vstack((data,none_matrix))
        elif n<m:
            none_matrix = np.array([empty]*col_num*(m-n))
            none_matrix = np.reshape(none_matrix, (m-n,11))
            data_new = np.vstack((data_new,none_matrix))
        data = np.hstack((data,data_new))

    return data


def setup_formatted_files(gt,ts_data,agent):
    # Get list of neighbours
    neighbours = []
    for key in ts_data[0][str(agent)]:
        neighbours = neighbours + [key]

    # Create csv file to store data for mult_twr=0
    with open("datasets/2022_02_21/formatted_ID"+str(agent)+"_twr0.csv", 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file -----------------------------------------------
        neighbour_headers = ["target_id", "mocap_ts", "gt",
                             "tx1", "rx1", "tx2", "rx2", "tx3", "rx3", None, None]
        row = ["self_id", None] + neighbour_headers*len(neighbours)
        writer.writerow(row)
    
        data = process_multitag_data(gt, ts_data, neighbours, agent, 0)
        data1 = data.astype(str)
        data1[data1=='nan'] = ''
        np.savetxt(f, data1, delimiter=",", fmt="%s")

    # Create csv file to store data for mult_twr=0
    with open("datasets/2022_02_21/formatted_ID"+str(agent)+"_twr1.csv", 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file -----------------------------------------------
        neighbour_headers = ["target_id", "mocap_ts", "gt",
                             "tx1", "rx1", "tx2", "rx2", "tx3", "rx3", None, None]
        row = ["self_id", None] + neighbour_headers*len(neighbours)
        writer.writerow(row)
    
        data = process_multitag_data(gt, ts_data, neighbours, agent, 1)
        data1 = data.astype(str)
        data1[data1=='nan'] = ''
        np.savetxt(f, data1, delimiter=",", fmt="%s")

    # Create csv file to store data for mult_twr=0
    with open("datasets/2022_02_21/formatted_ID"+str(agent)+"_twr2.csv", 'w') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file -----------------------------------------------
        neighbour_headers = ["target_id", "mocap_ts", "gt",
                             "tx1", "rx1", "tx2", "rx2", "tx3", "rx3", None, None]
        row = ["self_id", None] + neighbour_headers*len(neighbours)
        writer.writerow(row)
    
        data = process_multitag_data(gt, ts_data, neighbours, agent, 2)
        data1 = data.astype(str)
        data1[data1=='nan'] = ''
        np.savetxt(f, data1, delimiter=",", fmt="%s")


if __name__ == "__main__":
    mean_gt_distance = np.zeros((10,3))
    ts_data = [None] * 10
    mean_range_meas = np.zeros((10,3))
    for i in range(10):
        r = extract_gt_data(i+1)
        mean_gt_distance[i,:] = calculate_mean_gt_distance(r)
        temp_dict = {}
        temp_dict["1"] = extract_ts_data(i+1,1)
        temp_dict["2"] = extract_ts_data(i+1,2)
        ts_data[i] = temp_dict
        mean_range_meas[i,:] = calculate_mean_range(i+1,
                                                    ts_data[i]["1"],
                                                    ts_data[i]["2"],
                                                    2)
        
    print(mean_gt_distance)

    for i in range(10):
        mean_gt_distance[i,:] = match_gt_with_range(mean_gt_distance[i,:],
                                                    mean_range_meas[i,:])

    print(mean_gt_distance)

    setup_formatted_files(mean_gt_distance,ts_data,1)
    setup_formatted_files(mean_gt_distance,ts_data,2)


