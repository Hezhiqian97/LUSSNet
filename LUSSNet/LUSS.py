import colorsys
import copy
import time
from nets.LUSS import LUSS
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from utils.utils import cvtColor, preprocess_input, resize_image, show_config

class LUSS(object):
    _defaults = {


        "model_path": r'D:\SUAM_1\usam_v4\logs\drk\best_epoch_weights.pth',

        "num_classes": 8,

        "input_shape": [640, 640],

        "mix_type": 0,

        "cuda": True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128),
                           (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

        show_config(**self._defaults)


    def generate(self, onnx=False):
        self.net = LUSS(num_classes=self.num_classes)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()


    def detect_image(self, image, count=False, name_classes=None):

        image = cvtColor(image)

        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]

            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            pr = pr.argmax(axis=-1)

        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:

            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            image = Image.fromarray(np.uint8(seg_img))

            image = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:

            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])

            image = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')

            image = Image.fromarray(np.uint8(seg_img))

        return image

    def get_FPS(self, image, test_interval):
        image = cvtColor(image)
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                self.net.eval()
                print(next(self.net.parameters()).device)
                print(images.device)

                print(images.size())


            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            if self.cuda:
                torch.cuda.synchronize()

            t1 = time.time()

            for _ in range(test_interval):
                pr = self.net(images)[0]


            if self.cuda:
                torch.cuda.synchronize()
            t2 = time.time()

            tact_time = (t2 - t1) / test_interval
            return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)


        model_onnx = onnx.load(model_path)
        onnx.checker.check_model(model_onnx)

        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_miou_png(self, image):

        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]

            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
