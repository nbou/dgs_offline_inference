from offline_inf import InfConfig, Inference, generate_sys_cfg
from tkinter import filedialog, simpledialog
import tkinter as tk
import os
import configparser

if __name__ == '__main__':
    # first check if the system config file exists at ~/system_config.ini
    if not os.path.exists(os.path.expanduser('~/system_config.ini')):
        generate_sys_cfg()
    else:
        print("System config file found at ~/system_config.ini")

    # load the system config info
    config = configparser.ConfigParser()
    config.read(os.path.expanduser('~/system_config.ini'))
    model_path = config['Paths']['model_path']
    external_drive_path = config['Paths']['external_drive_path']
    results_dir = config['Paths']['results_dir']
    input_shape = eval(config['Paths']['input_shape'])
    input_layer_name = config['Paths']['input_layer_name']
    gpu = config.getboolean('Paths', 'gpu')

    # Then prompt the user for the run info
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    survey_dir = filedialog.askdirectory(
        title="Select Survey Directory",
        initialdir=external_drive_path
    )
    cam_number = simpledialog.askinteger("Camera Number", "Select Camera Number (1 or 2):", minvalue=1, maxvalue=2)
    results_dir = os.path.join(results_dir, os.path.basename(survey_dir))
    # os.makedirs(results_dir, exist_ok=True)


    nth_image = simpledialog.askinteger("Nth Image", "Process every nth image (e.g., 1 for every image, 5 for every 5th image):", minvalue=1, initialvalue=1)

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
