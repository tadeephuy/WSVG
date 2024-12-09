from selfeq_model import SelfEQ
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from pathlib import Path
from torch.nn import functional as F
from matplotlib import pyplot as plt
selfeq = SelfEQ()
size = 224
device = 'cuda'

img_model = './img_model.pt'
txt_model = './txt_model.pt'

selfeq.load(img_p=img_model, txt_p=txt_model)

selfeq = selfeq.to(device)
transform = create_chest_xray_transform_for_inference(size,size)


selfeq.inference_vg(
    'consolidation',
    './ms_2.png',
    device=device,
    viz=True
)

plt.axis('off')
plt.show()