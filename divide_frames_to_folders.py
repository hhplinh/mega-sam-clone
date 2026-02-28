import os
import glob
import argparse
import shutil

def divide_frames_to_folders(input_folder, output_folder, n_frames_per_folder):
    if n_frames_per_folder <= 0:
        raise ValueError("Number of frames per folder must be greater than 0.")
    if input_folder == output_folder:
        raise ValueError("Input and output folders must be different.")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    for i in range(0, len(frame_files), n_frames_per_folder):
        folder_name = os.path.join(output_folder, f'folder_{i // n_frames_per_folder + 1}')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for frame_file in frame_files[i:i + n_frames_per_folder]:
            shutil.copy2(frame_file, os.path.join(folder_name, os.path.basename(frame_file)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Divide frames into folders.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing frames.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder where divided folders will be created.')
    parser.add_argument('--n_frames_per_folder', type=int, required=True, help='Number of frames per folder.')

    args = parser.parse_args()
    divide_frames_to_folders(args.input_folder, args.output_folder, args.n_frames_per_folder)

# python divide_frames_to_folders.py --input_folder inference/data_test --output_folder inference/split_data --n_frames_per_folder 6