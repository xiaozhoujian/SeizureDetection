import torch
import os
import numpy as np
from PIL import Image
from clip_generation_gradcam import generate_clip
from models.resnext import resnext101
from misc_functions import apply_colormap_on_image, format_np_output
from CAM_opts import parse_opts


class CamExtractor:
    """
        Extracts cam features from the model
    """

    def __init__(self, model, opts, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.opts = opts

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.module._modules.items():
            if module_pos == "fc":
                break
            x = module(x)  # Forward
            if module_pos == "layer4":
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, feature_x = self.forward_pass_on_convolutions(x)
        feature_x = feature_x.view(feature_x.size(0), -1)
        # Forward pass on the classifier
        out = self.model.module.fc(feature_x)
        return conv_output, out


class GradCam:
    """
        Produces class activation map
    """

    def __init__(self, model, opts, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, opts, target_layer)

    def generate_cam(self, clip, opts):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(clip)
        if opts.clip_class is None:
            opts.clip_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][opts.clip_class] = 1
        # Zero grads
        self.model.module.zero_grad()
        # Backward pass with specified target
        one_hot_output = one_hot_output.cuda()
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.cpu().data.numpy()[0]
        # Get convolution outputs

        target = conv_output.cpu().data.numpy()[0][:, 0, :, :]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2, 3))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((clip.shape[3],
                                                    clip.shape[4]), Image.ANTIALIAS)) / 255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.
        return cam


def save_class_activation_images(org_img, activation_map, file_name, video_path, path_to_dir, opts):
    """
        Saves cam activation map and activation map on the original image
    :param org_img: PIL image, original image
    :param activation_map: numpy array, activation map(grayscale) 0-255
    :param file_name: str, file name of the exported image
    :param video_path: str, relative path of the video under the directory `path_to_dir`
    :param path_to_dir: str, directory to save the image files
    :param opts: ArgumentParser, contains config options
    """
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
    path_to_dir_cam = os.path.join(path_to_dir, video_path)
    if not os.path.exists(path_to_dir_cam):
        os.makedirs(path_to_dir_cam)
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, opts.cmaps)
    # Save colored heatmap
    path_to_file = os.path.join(path_to_dir_cam, file_name + "_Cam_Heatmap.png")
    save_image(heatmap, path_to_file)
    # Save heatmap on image
    path_to_file = os.path.join(path_to_dir_cam, file_name + "_Cam_On_Image.png")
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join(path_to_dir_cam, file_name + "_Cam_Grayscale.png")
    save_image(activation_map, path_to_file)


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    :param im: numpy array, matrix of shape DxWxH
    :param path: str, path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def main():
    # input_dir /root/5T/dataset/frame_MICCAI_dataset/control
    opts = parse_opts()
    # opts.input_path = "/dockerdata/yujiazhang/dataset/UCF_101_1f_256/{}/{}".format(class_name, opts.video_name)
    frame_path_list = []
    with open(opts.input_file, "r") as f:
        for line in f.readlines():
            frame_path_list.append(os.path.join(opts.frame_dir, line.strip()))

    pretrained_model_path = opts.pretrain_path
    pretrained_model = torch.load(pretrained_model_path)
    model = resnext101(num_classes=2, shortcut_type="B", cardinality=32, sample_size=112,
                       sample_duration=64, input_channels=3, output_layers=[])
    if opts.cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    else:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(pretrained_model["state_dict"], strict=True)

    for input_path in frame_path_list:  # [:2] Only for test
        opts.input_path = input_path
        v_path_split = input_path.split("/")
        if v_path_split[-2] == "control" or v_path_split[-2] == "pre_control":
            opts.clip_class = 0
        elif v_path_split[-2] == "case" or v_path_split[-2] == "pre_case":
            opts.clip_class = 1
        else:
            raise ValueError("Dataset Error")
        video_path = "{}/{}".format(v_path_split[-2], v_path_split[-1])

        clip, img_ori = generate_clip(opts)
        clip = clip.cuda()
        grad_cam = GradCam(model, opts, target_layer=None)
        cam = grad_cam.generate_cam(clip, opts)
        file_name_to_export = v_path_split[-1]
        # model_name = opts.pretrain_path.split("/")[-4]
        # using the middle frame as the original img
        save_class_activation_images(img_ori[int(0.5 * len(img_ori))], cam, file_name_to_export, video_path,
                                     opts.output_dir, opts)
        print("Grad cam completed")


if __name__ == "__main__":
    main()
