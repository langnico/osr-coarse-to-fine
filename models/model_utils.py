import timm


def get_model(args, wrapper_class=None, *args_, **kwargs):

    if args.model in ['timm_resnet50', 'timm_resnet50_norm_features']:

        if not hasattr(args, 'resnet50_pretrain'):
            args.resnet50_pretrain = 'scratch'

         # Get model
        if args.resnet50_pretrain == 'imagenet_timm':
            model = timm.create_model('resnet50', num_classes=args.num_classes, pretrained=True)
        elif args.resnet50_pretrain == 'scratch':
            model = timm.create_model('resnet50', num_classes=args.num_classes, pretrained=False)
        else:
            raise NotImplementedError

    elif args.model in ['wide_resnet50_2', 
                        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b7', 
                        'dpn92', 'resnet50', 'resnet18']:

        model = timm.create_model(args.model, num_classes=args.num_classes, pretrained=False)
    else:
        raise NotImplementedError

    if wrapper_class is not None:
        model = wrapper_class(model, *args_, **kwargs)

    return model

