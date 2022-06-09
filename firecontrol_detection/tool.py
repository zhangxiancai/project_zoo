def read_anchors():
    import torch
    from models.experimental import attempt_load

    model = attempt_load('/home/xiancai/fire-equipment-demo/firecontrol_detection/result/2021_11_30/best.pt', map_location=torch.device('cpu'))
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
    anc=m.anchor_grid.squeeze()
    print(anc)

read_anchors()
