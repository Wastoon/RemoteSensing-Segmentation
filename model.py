import torch
import segmentation_models_pytorch as smp


def get_model(data_channel=3, encoder=None, encoder_weight=None):
    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=encoder_weight,
        in_channels=data_channel,
        classes=10,
        encoder_depth=5,
    )
    return model

def test():
    input = torch.randn((1,3,256,256))
    model = get_model()
    print(model)
    output = model(input)
    print(output.shape)

if __name__=='__main__':
    test()
