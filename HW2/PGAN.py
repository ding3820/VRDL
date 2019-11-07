import torch
import matplotlib.pyplot as plt
import helper
import torchvision
import torch.nn.functional as F


def output_fig(images_array, file_name="./results"):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(helper.images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name + '.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':

    use_gpu = True if torch.cuda.is_available() else False

    # trained on high-quality celebrity faces "celebA" dataset
    # this model outputs 512 x 512 pixel images
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                           'PGAN', model_name='celebAHQ-512',
                           pretrained=True, useGPU=use_gpu)
    # this model outputs 256 x 256 pixel images
    # model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
    #                        'PGAN', model_name='celebAHQ-256',
    #                        pretrained=True, useGPU=use_gpu)

    for i in range(500):
        num_images = 9
        noise, _ = model.buildNoiseData(num_images)
        with torch.no_grad():
            generated_images = model.test(noise)

        generated_images = F.interpolate(generated_images, size=(112, 112), mode='bilinear', align_corners=False)
        print(generated_images.shape)   # should be (9, width, height, 3)
        grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), nrow=3, scale_each=True, normalize=True)

        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.savefig("images/{}_image".format(str.zfill(str(i), 3)) + '.png', bbox_inches='tight', pad_inches=0)




    # let's plot these images using torchvision and matplotlib

    # grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
    # plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    # plt.show()
