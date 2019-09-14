import os
import shutil

from GleasonChallenge2019.CustomerPath import divited_folder, all_data_folder, store_folder

def GetDivitedIndex(divited_folder, all_data_folder, store_folder):
    for cohort_index in ['Test', 'Train', 'Validation']:
        data_path = os.path.join(divited_folder, cohort_index)
        cohort_store_folder = os.path.join(store_folder, cohort_index)
        if not os.path.exists(cohort_store_folder):
            os.mkdir(cohort_store_folder)

        for reference_h5_file in os.listdir(data_path):
            original_h5_path = os.path.join(all_data_folder, reference_h5_file)
            store_h5_file = os.path.join(store_folder, cohort_index,reference_h5_file)
            shutil.copy(original_h5_path, store_h5_file)
            # print((os.path.join(data_path, reference_h5_file), original_h5_path, store_h5_file))


def main():
    GetDivitedIndex(divited_folder, all_data_folder, store_folder)

if __name__ == '__main__':
    main()