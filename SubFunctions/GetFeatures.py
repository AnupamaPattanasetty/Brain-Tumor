import cv2
import numpy as np
from dataclasses import dataclass
from keras.models import Model
from termcolor import cprint
import warnings
from SubFunctions.Attention import SqueezeExcitation, CBAM, EfficientChannel, AFF
from keras.applications.resnet import ResNet50, ResNet152
from SubFunctions.GLCM import GLCM_Features
from SubFunctions.StructuralPattern import StructuralPattern

resnet152 = ResNet152(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
resnet50 = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
warnings.filterwarnings("ignore", category=RuntimeWarning)



@dataclass
class FeatureExtraction(object):
    def __init__(self, image: np.ndarray):
        self.image = image



    @staticmethod
    def glcm_features(image) -> np.ndarray:
        cprint("[⚠️] GLCM features ", color='yellow', attrs=['bold'])
        GF = GLCM_Features()

        energy = GF.fast_glcm_ASM(image)[0]
        asm = GF.fast_glcm_ASM(image)[1]
        dissimilarity = GF.fast_glcm_dissimilarity(image)
        homogeneity = GF.fast_glcm_homogeneity(image)
        entropy = GF.fast_glcm_entropy(image)
        contrast = GF.fast_glcm_contrast(image)

        features = np.zeros(shape=(image.shape[0], image.shape[1], 6))
        features[:, :, 0] = energy
        features[:, :, 1] = asm
        features[:, :, 2] = dissimilarity
        features[:, :, 3] = homogeneity
        features[:, :, 4] = entropy
        features[:, :, 5] = contrast

        features = np.nan_to_num(features)

        return features


    @staticmethod
    def attentional_pretrained_features(image) -> np.ndarray:
        cprint("[⚠️] Distributed attentional fused pretrained model ", color='yellow', attrs=['bold'])

        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)

        x = resnet50.layers[2].output  # (None, 112, 112, 64)
        channels = x.shape[-1]
        x1 = SqueezeExcitation(num_channels=channels)(x)
        x2 = CBAM(channels=channels)(x)
        x3 = EfficientChannel(channels=channels, k_size=3)(x)

        x11 = AFF(channels=channels)(x1, x2)
        x = AFF(channels=channels)(x11, x3)
        # Build new model
        attention_resnet50 = Model(inputs=resnet50.input, outputs=x)

        outputs = np.squeeze(attention_resnet50.predict(image, verbose=0))
        outputs = np.mean(outputs, axis=-1)

        outputs = cv2.resize(outputs, (96, 96))
        return outputs

    @staticmethod
    def fused_structural_pattern(image) -> np.ndarray:
        cprint("[⚠️] Deep fused structural pattern ", color='yellow', attrs=['bold'])

        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)

        resnet_model = Model(inputs=resnet152.inputs, outputs=resnet152.layers[2].output)
        outputs = np.squeeze(resnet_model.predict(image, verbose=0))
        outputs = np.mean(outputs, axis=-1)
        outputs = cv2.resize(outputs, (96, 96))
        structural_pattern = StructuralPattern(outputs).get_structural_pattern()
        return structural_pattern

    @staticmethod
    def histogram(image, channel):
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
        return hist.flatten()




    def get_features1(self):
        output = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        output = output / 255.0
        output = cv2.resize(output, (32, 32))
        return output


    def get_features2(self):
        output = cv2.resize(self.image, (32, 32))
        output = output.astype(np.uint8)
        output = self.histogram(output, 0)
        return output


    def get_features3(self):
        output = cv2.resize(self.image, (32, 32))
        output = output.astype(np.uint8)
        output = self.histogram(output, 0)
        return output

    def get_features4(self):
        output = cv2.resize(self.image, (32, 32))
        output = output.astype(np.uint8)
        output = self.histogram(output, 0)
        return output


    def get_features5(self):
        output = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        output = output / 255.0
        output = cv2.resize(output, (32, 32))
        return output

    def get_features6(self):
        output = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        output = output / 255.0
        output = cv2.resize(output, (32, 32))
        return output

    def get_features7(self):
        output = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        output = output / 255.0
        output = cv2.resize(output, (32, 32))
        return output

    def get_features8(self):
        cprint('\n')
        cprint("================================", color='blue')
        cprint("[⚠️] Feature Extraction ", color='grey', on_color='on_yellow')
        cprint("================================", color='blue')

        pretrained_features = self.attentional_pretrained_features(self.image)
        glcm_features = self.glcm_features(self.image)
        structural_features = self.fused_structural_pattern(self.image)

        features = np.concatenate((np.expand_dims(pretrained_features, axis=-1),
                                   glcm_features,
                                   np.expand_dims(structural_features, axis=-1)), axis=-1)
        return features




