import torch
import numpy as np

def grad_cam(model, image_tensor, target_layer):
    # model eval initiation
    model.eval()

    # convert image to pytorch tensor
    #image_tensor = torch.from_numpy(image).float().unsqueeze(0)

    # activate hook to get output and gradient of target layer
    activations = None
    gradients = None

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out

    handle = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    # run forward
    output = model(image_tensor)
    idx = torch.argmax(output)

    # run backward
    model.zero_grad()
    output[0, idx].backward()

    # remove hooks
    handle.remove()
    handle_bw.remove()

    # calc grad cam
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze()
    cam = np.maximum(cam.detach().cpu().numpy(), 0)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

    return cam
