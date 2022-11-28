# Integrate heatmap and image
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import copy


def center_crop(img, crop_size=(112, 112)):
    w, h = img.size
    th, tw = crop_size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return img.crop((x1, y1, x1 + tw, y1 + th))


def apply_heatmap(ori_vid_path, frame_index, heatmap_path, out_dir=None, alpha=0.7):
    cap = cv2.VideoCapture(ori_vid_path)
    cap.set(1, frame_index)
    ret, ori_img = cap.read()
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(os.path.join(out_dir, "{}_src.png".format(ori_vid_path.strip().split("/")[-1].split(".mp4")[0])),
                ori_img)
    ori_img = Image.fromarray(ori_img)

    heatmap_img = copy.copy(Image.open(heatmap_path))
    crop_size = min(ori_img.size)
    heatmap_img = heatmap_img.resize((crop_size, crop_size))
    heatmap_img.putalpha(int(255 * alpha))
    # Apply heatmap on iamge
    ori_img = center_crop(ori_img, (crop_size, crop_size))
    heatmap_on_image = Image.new("RGBA", ori_img.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, ori_img.convert('RGBA'))

    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap_img)
    plt.imshow(heatmap_on_image, aspect='auto')
    if out_dir:
        png_path = os.path.join(out_dir, "{}.png".format(ori_vid_path.strip().split("/")[-1].split(".mp4")[0]))
        plt.savefig(png_path)
        # plt.show()


def main():
    # case
    my_dpi = 96
    # name2idx = {"2020-06-29 15-49-34__fps10_36": 288, "2019-10-26 17-47-59__fps10_59": 105, "2019-10-17 14-38-05__fps10_102":242}

    name2idx = {"2020-06-29 15-49-34__fps10_36": 288, "2019-10-26 17-47-59__fps10_59": 105,
                "2020-03-16 18-45-25__fps10_118": 115}
    for file_name in name2idx.keys():
        plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)

        heatmap_path = r"/root/5T/dataset/chamber_dataset/CAM_result_ft_large/pre_case/{}/{}_Cam_Heatmap.png".format(
            file_name, file_name)
        ori_vid_path = r"/root/5T/dataset/chamber_dataset/stephen_dataset/pre_case/{}.mp4".format(file_name)
        out_dir = "/root/5T/dataset/chamber_dataset/CAM_demo_ft_large/case"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        apply_heatmap(ori_vid_path, name2idx[file_name], heatmap_path, out_dir, alpha=0.6)


if __name__ == '__main__':
    main()
