# Code to run offline inference using ONNX Runtime

import onnxruntime as ort
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import os

# from torch.utils.tensorboard.summary import image_boxes
import pandas as pd
from datetime import datetime

class InfConfig:
    def __init__(self,
                 survey_dir:str,
                 model_path:str,
                 nth_image:int, # only inf every nth image
                 input_shape:tuple,
                 input_layer_name:str,
                 image_ext='.jpg',
                 gpu=False,
):
        self.survey_dir = survey_dir
        self.model_path = model_path
        self.nth_image = nth_image

        self.input_shape = input_shape
        self.data = pd.read_csv(os.path.join(survey_dir, 'photo_log.csv'))
        ts = 'inference_' + datetime.now().strftime('%Y%m%d-%H%M%S')

        self.results_dir = os.path.join(survey_dir, ts)
        os.makedirs(self.results_dir, exist_ok=True)
        self.input_layer_name = input_layer_name
        self.image_ext = image_ext
        self.gpu = gpu
        self.model = self.load_model(model_path)

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
        csvpth = os.path.join(self.config.results_dir, 'inference_results.csv')

        # df.to_csv(csvpth, index=False)

        # get the list of files to inference on
        files = self.config.data['filename_string']
        # extract every nth image (where n lives in self.config.nth_image)
        files = files[::self.config.nth_image]

        for file in files:
            image_path = os.path.join(self.config.survey_dir, file)
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
            print(len(newdf))
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

if __name__ == "__main__":
    cfg = InfConfig(
        survey_dir='../../data/sample_image_dir',
        model_path='../../data/model/Mobilenet-28-3-256-256.onnx',
        nth_image=4,
        input_shape=(256, 256),
        input_layer_name='input',
    )

    inf = Inference(cfg)
    results = inf.inf_directory()
    for img_name, output in results.items():
        print(f"Image: {img_name}, Output shape: {output.shape}")




