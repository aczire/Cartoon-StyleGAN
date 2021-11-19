import os
import torch
import matplotlib.pyplot as plt
import math

google_drive_paths = {

    "ffhq256.pt" : "https://drive.google.com/uc?id=1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO",
    "encoder_ffhq.pt" : "https://drive.google.com/uc?id=1QQuZGtHgD24Dn5E21Z2Ik25EPng58MoU",

    # Naver Webtoon
    "NaverWebtoon.pt" : "https://drive.google.com/uc?id=1wHNuRx5KwYcXnkfgwHPdEwXECVQrdH2c",
    "NaverWebtoon_FreezeSG.pt" : "https://drive.google.com/uc?id=1p0q5rjWTFZVD0oN4L4Cq-1Z5QrTzPA_q",
    "NaverWebtoon_StructureLoss.pt" : "https://drive.google.com/uc?id=1JeffrgcMv8OfPMU-nbK4iJXwmgU2yXpa",

    "Romance101.pt" : "https://drive.google.com/uc?id=1NVjyjyye7UQgTwyclPHx1R8dDOEt4b_U",

    "TrueBeauty.pt" : "https://drive.google.com/uc?id=1em9meBQ9YCnBQkQiiqH4HHVNobhZA6NL",

    "Disney.pt" : "https://drive.google.com/uc?id=1bc9Mh6cEzlkHZJwnRrPX6T2EGqSHh-6c",
    "Disney_FreezeSG.pt" : "https://drive.google.com/uc?id=1vQ0ZSjwgVkjxS9XojJF_khI50aME-Bap",
    "Disney_StructureLoss.pt" : "https://drive.google.com/uc?id=1_RIR8gGnadR0lGBlVSGQVCQ2AQZuQJnI",
    
    "Metface_FreezeSG.pt" : "https://drive.google.com/uc?id=1oAUZlE2BEmE8hCCBbq4kJIyQHousmTep",
    "Metface_StructureLoss.pt" : "https://drive.google.com/uc?id=1--_sfvGo5auSNwsf3lY78UyXaU6goA3y",
}

def download_pretrained_model(download_all=True, file=''):

    if not os.path.isdir('networks'):
        os.makedirs('networks')

    from gdown import download as drive_download
    
    if download_all==True:
        for nn in google_drive_paths:
            url = google_drive_paths[nn]
            networkfile = os.path.join('networks', nn)
            drive_download(url, networkfile, quiet=False)

    else:
        url = google_drive_paths[file]
        networkfile = os.path.join('networks', file)

        drive_download(url, networkfile, quiet=False)
        

# ---------------
# for styleclip

def ensure_checkpoint_exists(model_weights_filename):
    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename in google_drive_paths
    ):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename not in google_drive_paths
    ):
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights."
        )



# ========================================

def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

def image2tensor(image):
    image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)/255.
    return (image-0.5)/0.5

def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

def imshow(img, size=5, cmap='jet'):
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()

def save_image(img, size=5, out='output.png' , cmap='jet'):
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.savefig(out, dpi=300)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp
