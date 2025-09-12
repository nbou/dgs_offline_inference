# Modified script to run if you don't have a screen. Replaces the file dialogues with script arguements.

from offline_inf import InfConfig, Inference, generate_sys_cfg_nogui
# from tkinter import filedialog, simpledialog
#import tkinter as tk
import os
import configparser

from argparse import ArgumentParser

if __name__ == '__main__':
    # first check if the system config file exists at ~/system_config.ini
    if not os.path.exists('/media/data/PROCESSED_DATA/system_config.ini'):
        # ask user to provide the info via command line
        model_pth = input("Enter the full path to the model file: ")
        ext_drive_path = input("Enter the path to the external drive (e.g., /media/data): ")
        res_dir = input("Enter the path to the results directory: ")
        gpu = input("Use GPU? (yes/no): ").strip().lower() == 'yes'
        generate_sys_cfg_nogui(model_pth, ext_drive_path, res_dir, gpu=gpu)
    else:
        print("System config file found at PROCESSED_DATA/system_config.ini on the data drive")

    # load the system config info
    config = configparser.ConfigParser()
    config.read('/media/data/PROCESSED_DATA/system_config.ini')
    model_path = config['Paths']['model_path']
    external_drive_path = config['Paths']['external_drive_path']
    results_dir = config['Paths']['results_dir']
    input_shape = eval(config['Paths']['input_shape'])
    input_layer_name = config['Paths']['input_layer_name']
    gpu = config.getboolean('Paths', 'gpu')

    # Then prompt the user for the run info
 #   root = tk.Tk()
  #  root.withdraw()  # Hide the main window


    parser = ArgumentParser(description="Run inference on survey data without GUI")
    parser.add_argument("survey_dir", type=str, help="Path to the survey directory")
    parser.add_argument("cam_number", type=int, choices=[1, 2], help="Camera number (1 or 2)")
    parser.add_argument("--nth_image", type=int, default=1, help="Process every nth image (default: 1)")
    args = parser.parse_args()

    # survey_dir = filedialog.askdirectory(
    #     title="Select Survey Directory",
    #    initialdir=external_drive_path
    # )

    # cam_number = simpledialog.askinteger("Camera Number", "Select Camera Number (1 or 2):", minvalue=1, maxvalue=2)
    # results_dir = os.path.join(results_dir, os.path.basename(survey_dir))
    # os.makedirs(results_dir, exist_ok=True)


    # nth_image = simpledialog.askinteger("Nth Image", "Process every nth image (e.g., 1 for every image, 5 for every 5th image):", minvalue=1, initialvalue=1)

    survey_dir = args.survey_dir
    cam_number = args.cam_number
    results_dir = os.path.join(results_dir, os.path.basename(survey_dir))
    os.makedirs(results_dir, exist_ok=True)
    nth_image = args.nth_image

    # do the inference
    inf_cfg = InfConfig(
        survey_dir=survey_dir,
        cam_number=cam_number,
        model_path=model_path,
        nth_image=nth_image,
        input_layer_name=input_layer_name,
        input_shape=input_shape,
        gpu=gpu,
        results_dir=results_dir
    )
    # print config for confirmation
    print("Inference Configuration:")
    print(f"Survey Directory: {inf_cfg.survey_dir}")
    print(f"Camera Number: {inf_cfg.cam_number}")
    print(f"Model Path: {inf_cfg.model_path}")
    print(f"Process every {inf_cfg.nth_image} image(s)")
    print(f"Input Shape: {inf_cfg.input_shape}")
    print(f"Input Layer Name: {inf_cfg.input_layer_name}")
    print(f"Use GPU: {inf_cfg.gpu}")
    print(f"Results Directory: {inf_cfg.results_dir}")



    inference = Inference(inf_cfg)
    inference.inf_directory()