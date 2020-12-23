# download and process the natural questions dataset and save it to a path
from datasets import load_dataset
import argparse


if __name__ == '__main__':
    """Instructions
    
    to make changes to natural questions dataset please go the --config path and make changes to dataset_infos.json and 
    .py file in the folder
    
    also make sure you have enough space in the disk containing --path natural questions med take around 100G
    """
    parser = argparse.ArgumentParser(description='Download and prepare dataset')
    parser.add_argument('--config', '-c', default='../dataset_configs/natural_questions_small', type=str, help='path to data loader folder containing processing script and info')
    parser.add_argument('--path', '-p', default='../saved_datasets/natural_questions_small', type=str, help='location where the dataset is to be saved')
    args = parser.parse_args()

    # downloading and preparing dataset according to the data processing script in path
    dataset_downloaded = load_dataset(args.config, beam_runner='DirectRunner')

    # saving to disk for future loading purposes
    dataset_downloaded.save_to_disk(args.path)
