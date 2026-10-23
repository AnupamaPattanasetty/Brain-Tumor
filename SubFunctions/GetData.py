import dataclasses
import pickle
from termcolor import cprint
import glob
import nibabel as nib
import numpy as np
import tqdm
from SubFunctions.SegmentationGAN import SegmentationGAN
from SubFunctions.GetPreprocessing import Preprocessing
from SubFunctions.GetFeatures import FeatureExtraction
import os


@dataclasses.dataclass
class ReadDataset(object):
    def __init__(self, dataset: int, exec: bool = True):
        self.dataset = dataset
        self.exec = exec

    def read_data(self):
        if self.exec:
            cprint(f"[⁉️] Extracting Features and Labels from Dataset {self.dataset} ", color='grey',
                   on_color='on_white')

            # root_path = "C:\\CODE\\AnupamaP1"
            root_path = os.getcwd()

            if self.dataset == 1:

                path = glob.glob(
                    f"{root_path}\\DATASET\\Dataset1\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\*")

                modalities = {1: 't1',
                              2: 't1ce',
                              3: 't2',
                              4: 'flair'}

                # Initialize feature and label lists
                Features1, Features2, Features3, Features4 = [], [], [], []
                Features5, Features6, Features7, Features8 = [], [], [], []
                Labels = []
                for foldername in tqdm.tqdm(path, desc='Reading data'):

                    for modality in modalities.values():
                        file_path = foldername + "\\" + foldername.split("\\")[-1] + "_" + modality + ".nii"
                        mask_path = foldername + "\\" + foldername.split("\\")[-1] + "_seg.nii"
                        image_data = nib.load(file_path).get_fdata()

                        mask_data = nib.load(mask_path).get_fdata()

                        for i in tqdm.tqdm(range(image_data.shape[-1]), desc='Reading Slices'):
                            image = image_data[:, :, i]
                            mask = mask_data[:, :, i]

                            segmented = SegmentationGAN(image, mask).get_segmented()

                            preprocessed = Preprocessing(segmented).get_preprocessing()

                            # Extract multiple feature sets from the image
                            features1 = FeatureExtraction(preprocessed).get_features1()
                            features2 = FeatureExtraction(preprocessed).get_features2()
                            features3 = FeatureExtraction(preprocessed).get_features3()
                            features4 = FeatureExtraction(preprocessed).get_features4()
                            features5 = FeatureExtraction(preprocessed).get_features5()
                            features6 = FeatureExtraction(preprocessed).get_features6()
                            features7 = FeatureExtraction(preprocessed).get_features7()
                            features8 = FeatureExtraction(preprocessed).get_features8()

                            # Append features to respective lists
                            Features1.append(features1)
                            Features2.append(features2)
                            Features3.append(features3)
                            Features4.append(features4)
                            Features5.append(features5)
                            Features6.append(features6)
                            Features7.append(features7)
                            Features8.append(features8)

                            if len(np.unique(mask)) == 4:
                                Labels.append(3)
                            elif len(np.unique(mask)) == 3:
                                Labels.append(2)
                            elif len(np.unique(mask)) == 2:
                                Labels.append(1)
                            else:
                                Labels.append(0)



            else:
                path = glob.glob(f"{root_path}\\DATASET\\Dataset2\\BRATS2017\\Brats17TrainingData\\HGG\\*")

                modalities = {1: 't1',
                              2: 't1ce',
                              3: 't2',
                              4: 'flair'}

                # Initialize feature and label lists
                Features1, Features2, Features3, Features4 = [], [], [], []
                Features5, Features6, Features7, Features8 = [], [], [], []
                Labels = []

                for foldername in tqdm.tqdm(path, desc='Reading data'):

                    for modality in modalities.values():
                        file_path = foldername + "\\" + foldername.split("\\")[-1] + "_" + modality + ".nii"
                        mask_path = foldername + "\\" + foldername.split("\\")[-1] + "_seg.nii"
                        image_data = nib.load(file_path).get_fdata()

                        mask_data = nib.load(mask_path).get_fdata()

                        for i in tqdm.tqdm(range(image_data.shape[-1]), desc='Reading Slices'):

                            image = image_data[:, :, i]
                            mask = mask_data[:, :, i]

                            segmented = SegmentationGAN(image, mask).get_segmented(epochs=1)

                            preprocessed = Preprocessing(segmented).get_preprocessing()

                            # Extract multiple feature sets from the image
                            features1 = FeatureExtraction(preprocessed).get_features1()
                            features2 = FeatureExtraction(preprocessed).get_features2()
                            features3 = FeatureExtraction(preprocessed).get_features3()
                            features4 = FeatureExtraction(preprocessed).get_features4()
                            features5 = FeatureExtraction(preprocessed).get_features5()
                            features6 = FeatureExtraction(preprocessed).get_features6()
                            features7 = FeatureExtraction(preprocessed).get_features7()
                            features8 = FeatureExtraction(preprocessed).get_features8()

                            # Append features to respective lists
                            Features1.append(features1)
                            Features2.append(features2)
                            Features3.append(features3)
                            Features4.append(features4)
                            Features5.append(features5)
                            Features6.append(features6)
                            Features7.append(features7)
                            Features8.append(features8)

                            if len(np.unique(mask)) == 4:
                                Labels.append(3)
                            elif len(np.unique(mask)) == 3:
                                Labels.append(2)
                            elif len(np.unique(mask)) == 2:
                                Labels.append(1)

                            else:
                                Labels.append(0)

            # Convert all features and labels to NumPy arrays
            Features1 = np.array(Features1)
            Features2 = np.array(Features2)
            Features3 = np.array(Features3)
            Features4 = np.array(Features4)
            Features5 = np.array(Features5)
            Features6 = np.array(Features6)
            Features7 = np.array(Features7)
            Features8 = np.array(Features8)
            Labels = np.array(Labels).astype(int)

            # Pack features and labels into a dictionary
            data = {
                'Features1': Features1,
                'Features2': Features2,
                'Features3': Features3,
                'Features4': Features4,
                'Features5': Features5,
                'Features6': Features6,
                'Features7': Features7,
                'Features8': Features8,
                'Labels': Labels
            }

            if not os.path.exists(f'Features\\Dataset{self.dataset}'):
                os.makedirs(f'Features\\Dataset{self.dataset}')

            # Save the extracted features and labels as a pickle file for future use
            with open(f'Features\\Dataset{self.dataset}\\Features.pkl', 'wb') as f:
                pickle.dump(data, f)

            # Print confirmation message that feature extraction is complete
            cprint(f"[✅] Feature Extraction Done !! ", color='grey', on_color='on_white')

        else:
            # If features were already extracted earlier, load them from the pickle file
            cprint(f"[⁉️] Loading the Extracted Features and Labels ", color='grey', on_color='on_white')

            # Load the saved dictionary (features + labels) from the pickle file
            with open(f'Features\\Dataset{self.dataset}\\Features.pkl', 'rb') as f:
                data = pickle.load(f)

            # Print confirmation message that loading is complete
            cprint(f"[✅] Feature Loading Done !! ", color='grey', on_color='on_white')

        # Return the dictionary containing Features and Labels
        return data
