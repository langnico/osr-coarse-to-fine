import torch


def get_distance_matrix(train_feat, test_feat, dist_key, feat_norm=None):
    """Compute distance matrix between all train and test features

    Args:
        train_feat (torch.Tensor): training features (num_train, feat_dim)
        test_feat (torch.Tensor): test features (num_test, feat_dim)
        dist_key (str): distance metric key. Choices: ["L2", "cos"]
        feat_norm (str, optional): feature normalization key. Defaults to None. Choices: ["znorm", "l2norm", "unnorm"]

    Returns:
        distance matrix (torch.Tensor): (num_test, num_train)
    """

    # Normalize features
    if feat_norm == "znorm":
        # compute channel wise mean and std of training data
        train_mean = torch.mean(train_feat, dim=0)
        train_std = torch.std(train_feat, dim=0)

        train_feat = (train_feat - train_mean) / train_std
        test_feat = (test_feat - train_mean) / train_std

    elif feat_norm == "l2norm":
        train_feat = torch.nn.functional.normalize(train_feat, p=2, dim=1)
        test_feat = torch.nn.functional.normalize(test_feat, p=2, dim=1)
    elif feat_norm == "unnorm" or feat_norm is None:
        pass
    else:
        raise NotImplementedError

    # Compute distance matrix between all train and test features
    if dist_key == "L2":
        dists = torch.linalg.vector_norm(test_feat.unsqueeze(1) - train_feat.unsqueeze(0), ord=2, dim=2)
    
    elif dist_key == "cos":
        # cosine_distance = 1 - cosine_similarity
        dists = 1 - torch.nn.functional.cosine_similarity(test_feat[:,:,None], train_feat.t()[None,:,:])
    else:
        raise NotImplementedError
    
    return dists

