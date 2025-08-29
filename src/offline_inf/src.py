# Code to run offline inference using ONNX Runtime

import onnxruntime as ort
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import os

import pandas as pd
from datetime import datetime

# script to generate system config which contains the path to the model,
# where the external drive is mounted and where the results will be stored

import configparser
from tkinter import filedialog
import tkinter as tk

def generate_sys_cfg():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("ONNX files", "*.onnx"), ("All files", "*.*")])
    external_drive_path = filedialog.askdirectory(title="Select External Drive Directory", initialdir='/media/data/RAW_DATA')
    results_dir = filedialog.askdirectory(title="Select Results Directory", initialdir='/media/data/PROCESSED_DATA')
    gpu = tk.messagebox.askyesno("GPU Usage", "Do you want to use GPU for inference if available?")
    config = configparser.ConfigParser()
    config['Paths'] = {
        'model_path': model_path,
        'external_drive_path': external_drive_path,
        'results_dir': results_dir,
        'input_shape' : (256, 256),  # assuming a standard input shape, modify as needed
        'input_layer_name' : 'input',  # modify as needed
        'gpu': str(gpu)
    }

    with open('/media/data/PROCESSED_DATA/system_config.ini', 'w') as configfile:
        config.write(configfile)

    print("Configuration saved to PROCESSED_DATA/system_config.ini")

class InfConfig:
    def __init__(self,
                 survey_dir:str,
                 cam_number:int,
                 model_path:str,
                 nth_image:int, # only inf every nth image
                 input_shape:tuple,
                 input_layer_name:str,
                 gpu=False,
                 results_dir=None
):

        self.survey_dir = survey_dir
        if cam_number not in [1,2]:
            raise ValueError("cam_number must be 1 or 2")
        self.cam_number = cam_number
        self.model_path = model_path
        self.nth_image = nth_image

        self.input_shape = input_shape
        self.data = pd.read_csv(os.path.join(survey_dir, 'cam_{}'.format(self.cam_number), 'photo_log.csv'))
        self.image_dir = os.path.join(survey_dir, 'cam_{}'.format(self.cam_number))
        ts = 'inference_' + datetime.now().strftime('%Y%m%d-%H%M%S')

        # self.results_dir = os.path.join(survey_dir, ts)
        # os.makedirs(self.results_dir, exist_ok=True)
        self.input_layer_name = input_layer_name
        self.gpu = gpu
        self.model = self.load_model(model_path)
        self.results_dir = results_dir

    def load_model(self, model_path):
        if self.gpu:
            ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
            self.torch_device= torch.device('cuda')
        else:
            ort_session = ort.InferenceSession(model_path)
            self.torch_device = torch.device('cpu')
        return ort_session

class Inference:
    def __init__(self, config:InfConfig):
        self.config = config
        self.session = config.model
        self.results_dir = self.config.results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # self.config.results_dir = self.results_dir
        # write config to results dir for record keeping
        with open(os.path.join(self.results_dir, 'inference_config.txt'), 'w') as f:
            for key, value in self.config.__dict__.items():
                if key != 'model' or key != 'data':  # don't write the model or data to file
                    f.write(f"{key}: {value}\n")

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        grid, _, _ = img_to_grid(image, 4, 7)

        all_patches = []

        for patch in grid:
            patch = np.transpose(patch, (2, 0, 1))  # (C, H, W)
            patch_tensor = torch.from_numpy(patch)  # still on CPU
            patch_tensor = cropper(patch_tensor, 758, 760)  # (C, h, w)

            patch_tensor = patch_tensor.to(self.config.torch_device)

            # Resize using torchvision (to GPU directly)
            patch_tensor = TF.resize(patch_tensor, [256, 256], interpolation=TF.InterpolationMode.BICUBIC,
                                     antialias=True)

            # Normalize
            patch_tensor = patch_tensor.float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.config.torch_device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.config.torch_device).view(3, 1, 1)
            patch_tensor = (patch_tensor - mean) / std

            all_patches.append(patch_tensor.cpu().numpy())

        reshaped = np.asarray(all_patches)
        reshaped = reshaped.astype(np.float32)
        return reshaped

    def run_inference(self, input_data):
        input_name = self.config.input_layer_name
        # input_data = input_data.reshape(self.config.input_shape)
        inputs = {input_name: input_data}
        outputs = self.session.run(None, inputs)
        return outputs[0]  # Assuming the first output is the desired one

    def inf_directory(self):
        # run inference on all images in the directory. After every batch, update csv with the predictions for each
        # in the image
        results = {}
        # create a csv file to store the results, columns are image_name, patch_number, prediction
        # df = pd.DataFrame(columns=['image_name', 'patch_number', 'prediction'])
        csvpth = os.path.join(self.results_dir, 'inference_results.csv')

        # df.to_csv(csvpth, index=False)

        # get the list of files to inference on
        files = self.config.data['filename_string']
        # extract every nth image (where n lives in self.config.nth_image)
        files = files[::self.config.nth_image]

        for file in files:
            image_path = os.path.join(self.config.image_dir, file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read image {image_path}. Skipping.")
                continue
            input_data = self.preprocess(image)
            output = self.run_inference(input_data)
            results[file] = output
            # Append results to csv
            newdf = []
            for i, pred in enumerate(output):
                new_row = {'image_name': file, 'patch_number': i, 'prediction': pred}
                newdf.append(new_row) #pd.concat([newdf, pd.DataFrame([new_row])], ignore_index=True)
            pd.DataFrame(newdf).to_csv(csvpth, index=False, mode='a', header=not os.path.exists(csvpth))
            print(f"Processed {file}")
        return results
        #
        # for root, _, files in os.walk(self.config.survey_dir):
        #     for file in files:
        #         if file.endswith(self.config.image_ext):
        #             image_path = os.path.join(root, file)
        #             image = cv2.imread(image_path)
        #             if image is None:
        #                 print(f"Warning: Unable to read image {image_path}. Skipping.")
        #                 continue
        #             input_data = self.preprocess(image)
        #             output = self.run_inference(input_data)
        #             results[file] = output
        #             # Append results to csv
        #             for i, pred in enumerate(output):
        #                 new_row = {'image_name': file, 'patch_number': i, 'prediction': pred}
        #                 df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        #             df.to_csv(csvpth, index=False, mode='a',header=not os.path.exists(csvpth))
        #             print(f"Processed {file}")
        # return results

def img_to_grid(img, row, col):
    ww = [[i.min(), i.max()] for i in np.array_split(range(img.shape[0]), row)]
    hh = [[i.min(), i.max()] for i in np.array_split(range(img.shape[1]), col)]
    grid = [img[j:jj+1, i:ii+1, :] for j, jj in ww for i, ii in hh]
    return grid, len(ww), len(hh)

def cropper(images, width, height):
    C, H, W = images.shape
    start_x = (W - width) // 2
    start_y = (H - height) // 2
    return images[:, start_y:start_y + height, start_x:start_x + width]

# if __name__ == "__main__":
#     cfg = InfConfig(
#         survey_dir='../../data/sample_image_dir',
#         model_path='../../data/model/Mobilenet-28-3-256-256.onnx',
#         nth_image=4,
#         input_shape=(256, 256),
#         input_layer_name='input',
#     )
#
#     inf = Inference(cfg)
#     results = inf.inf_directory()
#     for img_name, output in results.items():
#         print(f"Image: {img_name}, Output shape: {output.shape}")




