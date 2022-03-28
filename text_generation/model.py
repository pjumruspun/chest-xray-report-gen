import torch
import torchvision
from torch import nn
from utils import CONDITIONS

# FINETUNED_WEIGHT_PATH = 'weights/pretrained_encoder/pretrained_enc_2022-02-26_16-38-13.950274.pth.tar'
FINETUNED_WEIGHT_PATH = 'weights/pretrained_encoder/pretrained_enc_epoch_5_2022-03-08_15-43-47.540586.pth.tar' # Full data 14

class Chexnet(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, input_size=(256, 256), pretrained=True):
        super(Chexnet, self).__init__()

        encoded_image_size = input_size[0] // 32
        self.linear_input_size = 1024 * encoded_image_size ** 2
        
        # Load densenet pretrained weight into this
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        
        # Original chexnet classifier
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 14),
            nn.Sigmoid()
        )

        # Pretrained weights?
        if pretrained:
            print(f"loading pretrained weights..")
            chexnet_checkpoint_path = 'weights/chexnet.pth.tar'
            checkpoint = torch.load(chexnet_checkpoint_path)
            self.load_state_dict(checkpoint['state_dict'], strict=False)

        # Create the true densenet we're going to use
        modules = list(self.densenet121.children())[:-1]
        self.true_densenet = nn.Sequential(*modules)

        # New classifier head
        self.fc = nn.Linear(self.linear_input_size, len(CONDITIONS))

        self = self.cuda()

    def forward(self, x):
        encoded_images = self.true_densenet(x)
        flattened = encoded_images.reshape(-1, self.linear_input_size)
        probs = torch.sigmoid(self.fc(flattened))
        return encoded_images, probs

    @staticmethod
    def finetuned():
        raise Exception("Please check the weights to proper path")
        print(f"loading weights from {FINETUNED_WEIGHT_PATH}")
        checkpoint = torch.load(FINETUNED_WEIGHT_PATH)
        return checkpoint['encoder'].cuda()