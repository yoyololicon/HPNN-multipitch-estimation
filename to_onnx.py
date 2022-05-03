import torch
from models import MLC_CFP_pianoroll

model = MLC_CFP_pianoroll(8192, 44100, [1] * 6, 512, 25, 17)

pretrain = torch.load('Models/musicnet_aug_6L.pth')

model.load_state_dict(pretrain.state_dict(), False)
jitted = torch.jit.script(model)
print(jitted)
jitted.save('Models/musicnet_aug_6L_jit.pt')

jitted = torch.jit.load('Models/musicnet_aug_6L_jit.pt')

x = torch.randn(1, 44100)

with torch.no_grad():
    y1 = model(x)
    y2 = jitted(x)

print(y1.shape, y2.shape)
print(torch.allclose(y1, y2))


# torch.onnx.export(jitted, x, 'musicnet_aug_6L.onnx')

# y = jitted(torch.randn(1, 16384))
# print(y.shape)
