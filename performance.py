import pickle
import numpy as np

if __name__ == "__main__":
    eval_path = "experiment/Qwen2-VL-7B-Instruct.pkl"
    with open(eval_path, 'rb') as f:
        data = pickle.load(f)
    result_raw = {}
    for dataset_name, dataset_value in data.items():
        result_raw[dataset_name] = np.mean(dataset_value['acc_list'])

    result_raw['perception_embody_result'] = (result_raw['perception1_embody_result'] + result_raw[
        'perception2_embody_result']) / 2
    del result_raw['perception1_embody_result']
    del result_raw['perception2_embody_result']
    data_name_list = ['perception_web_result', 'perception_embody_result', 'planning_web_result',
                      'planning_embody_result',
                      'planning_travel_result', 'safety_web_result', 'safety_embody_result']
    acc_list = []
    for data_name in data_name_list:
        acc_list.append(round(result_raw[data_name] * 100, 1))
    final_list = []
    final_list.append(acc_list[0])
    final_list.append(acc_list[1])
    final_list.append(round((acc_list[0] + acc_list[1]) / 2, 1))
    final_list.append(acc_list[2])
    final_list.append(acc_list[3])
    final_list.append(acc_list[4])
    final_list.append(round((acc_list[2] + acc_list[3] + acc_list[4]) / 3, 1))
    final_list.append(acc_list[5])
    final_list.append(acc_list[6])
    final_list.append(round((acc_list[5] + acc_list[6]) / 2, 1))
    final_list.append(
        round((acc_list[0] + acc_list[1] + acc_list[2] + acc_list[3] + acc_list[4] + acc_list[5] + acc_list[6]) / 7, 1))

    print("eval_path:", eval_path)
    print("Perception Web:", final_list[0])
    print("Perception Emb", final_list[1])
    print("Perception Avg.", final_list[2])

    print("Planning Web", final_list[3])
    print("Planning Emb", final_list[4])
    print("Planning Travel", final_list[5])
    print("Planning Avg.", final_list[6])

    print("Safety Web", final_list[7])
    print("Safety Emb", final_list[8])
    print("Safety Avg.", final_list[9])

    print("Total Avg.", final_list[10])
