def weight_transfer(model1, model2):
    """To overcome the slow converge speed in DETR model, use the weights in the 
    pretrained RT-DETR object detection model for the backbone and encoder.

    Parameters
    ----------
    model1 : The modified pose DETR model
    model2 : The pretrained RT-DETR model

    Returns
    -------
    THe modified pose DETR model with pretrained object detection model weight
    """
    pretrained_model_dict = model2.state_dict()
    new_model_dict = model1.state_dict()
    a= {}

    # This is the coco pretrained model weight
    # for k, v in pretrained_model_dict.items():
    #     if k in new_model_dict and 'model.28' not in k:
    #         a[k] = v

    # This is the green onion pretrained model weight
    for k, v in pretrained_model_dict.items():
        if k in new_model_dict:
            a[k] = v

    new_model_dict.update(a)
    model1.load_state_dict(new_model_dict)
    return model1