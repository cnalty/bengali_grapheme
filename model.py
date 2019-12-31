from torch import nn
import torchvision.models as models

class CharNet():
    def __init__(self):
        self.model = models.resnet34(pretrained=False)

def main():
    net = CharNet()
    print(net.model)

if __name__ == "__main__":
    main()


