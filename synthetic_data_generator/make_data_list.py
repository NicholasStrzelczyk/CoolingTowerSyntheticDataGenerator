import sys

import pandas as pd
from tqdm import tqdm

from synthetic_data_generator.utils.constants import *

if __name__ == '__main__':
    # ----- hyperparameters ----- #
    part = Partition.TRAIN

    #  ----- get data paths depending on platform ----- #
    if sys.platform == 'darwin':  # mac
        data_dir = "/Users/nick_1/Bell_5G_Data/synth_datasets/{}".format(part.value)
        list_name = "list_{}.txt".format(sys.platform)
    else:  # windows
        data_dir = "C:\\Users\\NickS\\UWO_Summer_Research\\Bell_5G_Data\\synth_datasets\\{}".format(part.value)
        list_name = "list_{}.txt".format(sys.platform)

    # ----- begin generating list ----- #
    list_file = open(os.path.join(data_dir, list_name), "a")

    for scenario in tqdm(range(1, 5), desc='Generating data lists'):
        scenario_dir = os.path.join(data_dir, "scenario_{}".format(scenario))
        df = pd.read_csv(os.path.join(scenario_dir, "timeline.csv"))
        total_days = df['day'].values[-1]

        for day in range(1, total_days + 1):
            label_name = "LABEL_day_{}.png".format(day)

            for hour in hour_list:
                image_name = "SYNTH_day_{}_{}.png".format(day, scenario)

                if os.path.exists(os.path.join(data_dir, image_name)):  # skip over missing hours
                    img_path = os.path.join(scenario_dir, "images", image_name)
                    tgt_path = os.path.join(scenario_dir, "targets", label_name)
                    list_file.write(img_path + " " + tgt_path + "\n")

    list_file.close()
