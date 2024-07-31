import sys

import pandas as pd
from tqdm import tqdm

from synthetic_data_generator.synth_data_utils.constants import *

if __name__ == '__main__':
    # ----- hyperparameters ----- #
    part = Partition.TEST

    #  ----- get data paths depending on platform ----- #
    if sys.platform == 'darwin':  # mac
        data_dir = "/Users/nick_1/Bell_5G_Data/synth_datasets/{}".format(part.value)
        list_name = "list_{}.txt".format(sys.platform)
    else:  # windows
        data_dir = "C:\\Users\\NickS\\UWO_Summer_Research\\Bell_5G_Data\\synth_datasets\\{}".format(part.value)
        list_name = "list_{}.txt".format(sys.platform)

    list_file_path = os.path.join(data_dir, list_name)
    open(list_file_path, 'w+').close()  # overwrite/ make new blank file

    # ----- begin generating list ----- #
    list_file = open(list_file_path, "a")

    for scenario in tqdm(range(1, 5), desc='Generating data lists'):
        scenario_dir = os.path.join(data_dir, "scenario_{}".format(scenario))
        total_days = pd.read_csv(os.path.join(scenario_dir, "timeline.csv"))['day'].values[-1]

        for day in range(1, total_days + 1):
            label_name = "LABEL_day_{}.png".format(day)

            for hour in hour_list:
                image_name = "SYNTH_day_{}_{}.png".format(day, hour)

                if os.path.exists(os.path.join(scenario_dir, "images", image_name)):  # skip over missing hours
                    img_path = os.path.join(scenario_dir, "images", image_name)
                    tgt_path = os.path.join(scenario_dir, "targets", label_name)
                    list_file.write(img_path + " " + tgt_path + "\n")

    list_file.close()
