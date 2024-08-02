import os
import json
import torch
import argparse
import numpy as np
import collections
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from core import evaluation
from core.knn import get_distance_matrix


parser = argparse.ArgumentParser("CollectEnsemble")

parser.add_argument('--dataset_train', type=str, default='aves', help="")
parser.add_argument('--model_path', type=str, help="")
parser.add_argument('--num_models', type=int, default=5, help="Number of models in ensemble.")
parser.add_argument('--num_models_pool', type=int, default=10, help="Total number of models available for selection.")
parser.add_argument("--test_hops", default=list(range(1, 8)), nargs='+', type=int,
                    help="List of hops being tested.")

# to select a random subset of model ids
parser.add_argument('--select_random_ids', action='store_true', default=False, help="")  
parser.add_argument('--ensemble_id', type=int, default=0, help="")



def init_nested_dict():
    return collections.defaultdict(init_nested_dict)


def get_model_paths(model_path, model_ids):
    return [
        os.path.join(model_path, "model_{}".format(model_id))
        for model_id in model_ids
    ]


def load_member_test_metrics(model_paths, dataset_train, dataset_test_list):
    metrics_dict = init_nested_dict()
    for model_id, path in enumerate(model_paths):
        for dataset_test in dataset_test_list:
            with open(
                os.path.join(path, "test", dataset_test, "metrics.json"), "r"
            ) as f:
                metrics_dict[dataset_train][dataset_test][model_id] = json.load(f)
    return metrics_dict


def load_member_model_outputs(model_paths, dataset_train, dataset_test_list):
    data_dict = init_nested_dict()
    for model_id, path in enumerate(model_paths):
        for dataset_test in dataset_test_list:
            data_dict[dataset_train][dataset_test][model_id] = dict(
                np.load(os.path.join(path, "test", dataset_test, "model_outputs.npz"), allow_pickle=True)
            )
    return data_dict


def compute_member_test_metrics_stats(
    metrics_dict, stats_fun_dict, num_models, dataset_train, dataset_test_list
):
    metrics_dict_member_stats = init_nested_dict()
    metric_keys = None
    for dataset_test in dataset_test_list:
        if metric_keys is None:
            # get list of metrics keys
            metric_keys = list(
                metrics_dict[dataset_train][dataset_test][0].keys()
            )  # using model_id=0
            if "Tax-Pool" in metric_keys:
                metric_keys.remove("Tax-Pool")
            if "Multi-Tax-Pred" in metric_keys:
                metric_keys.remove("Multi-Tax-Pred")

        for metric_key in metric_keys:
            # list of metric for each member
            metric_per_member = np.array(
                [
                    metrics_dict[dataset_train][dataset_test][model_id][metric_key]
                    for model_id in range(num_models)
                ]
            )
            for stats_key, stats_fun in stats_fun_dict.items():
                metrics_dict_member_stats[dataset_train][dataset_test]["members"][
                    metric_key
                ][stats_key] = float(stats_fun(metric_per_member))
    return metrics_dict_member_stats


def compute_ensemble_model_output(
    data_dict, num_models, dataset_train, dataset_test_list
):
    data_dict_ensemble = init_nested_dict()
    output_keys = None
    for dataset_test in dataset_test_list:
        if output_keys is None:
            output_keys = list(data_dict[dataset_train][dataset_test][0].keys())
            if "labels" in output_keys:
                output_keys.remove("labels")
            if "labels_u" in output_keys:
                output_keys.remove("labels_u")
        for key in output_keys:
            # stack member outputs along new axis=0
            data_key_stacked = np.stack(
                [
                    data_dict[dataset_train][dataset_test][model_id][key]
                    for model_id in range(num_models)
                ]
            )
            # reduce member outputs to get ensemble output
            data_key_averaged = np.mean(data_key_stacked, axis=0)
            data_dict_ensemble[dataset_train][dataset_test]["ensemble"][
                key
            ] = data_key_averaged
        # copy labels from model_id=0
        for label_key in ["labels", "labels_u"]:
            data_dict_ensemble[dataset_train][dataset_test]["ensemble"][label_key] = data_dict[
            dataset_train
            ][dataset_test][0][label_key]
    return data_dict_ensemble


def load_member_model_outputs_train(model_paths, dataset_train, output_keys=["feat_k"]):
    data_dict = init_nested_dict()
    for model_id, path in enumerate(model_paths):
        for key in output_keys:
            data_dict[dataset_train][model_id][key] = dict(
                np.load(os.path.join(path, "test", "model_outputs_train.npz"), allow_pickle=True)
            )[key]
    return data_dict


def compute_ensemble_model_output_train(
    data_dict, num_models, dataset_train, output_keys=["feat_k"]
):
    data_dict_ensemble = init_nested_dict()
    for key in output_keys:
        # stack member outputs along new axis=0
        data_key_stacked = np.stack(
            [data_dict[dataset_train][model_id][key] for model_id in range(num_models)]
        )
        # reduce member outputs to get ensemble output
        data_key_averaged = np.mean(data_key_stacked, axis=0)
        data_dict_ensemble[dataset_train]["ensemble"][key] = data_key_averaged
    # copy labels from model_id=0
    data_dict_ensemble[dataset_train]["ensemble"]["labels"] = data_dict[dataset_train][0]["labels"]
    return data_dict_ensemble


# Save results metrics dict
def save_metrics_to_json(d, filepath):
    with open(filepath, "w") as f:
        json.dump(d, f, indent=2)


def save_model_outputs_to_npz(d, filepath):
    np.savez(file=filepath, **d)


def KL_disagreement(ensemble_softmax, members_softmax):
    """
    KLD disagreement: The sum of KLD between the averaged ensemble output softmax and each individual model.
    :param ensemble_softmax: shape=(1, classes)
    :param members_softmax: shape=(samples, classes)
    :return: scalar

    We use the relative entropy (Kullback-Leibler divergence, KLD)
    https://stackoverflow.com/questions/57134984/compute-kl-divergence-between-rows-of-a-matrix-and-a-vector
    D = sum(pk * log(pk / qk))
    pk is the "true", i.e. the ensemble output;
    qk is the "prediction", i.e. the output of an individual member.
    """
    assert ensemble_softmax.shape[0] == 1
    KLD_ensemble_to_members = entropy(pk=ensemble_softmax, qk=members_softmax, axis=1)
    return float(np.sum(KLD_ensemble_to_members))


def compute_KLD_disagreement(data_dict_ensemble, data_dict_members, num_models):
    score_dict = {}
    for key, output_key in zip(
        ["known", "unknown"], ["preds_k_probs", "preds_u_probs"]
    ):
        score_dict[key] = []
        # for every sample: compute the sum of KLD(ensemble, model_i) over all i.
        for sample_id in range(len(data_dict_ensemble[output_key])):
            ensemble_softmax = data_dict_ensemble[output_key][sample_id, :][None, :]
            members_softmax = np.array(
                [
                    data_dict_members[model_id][output_key][sample_id, :]
                    for model_id in range(num_models)
                ]
            )
            score_dict[key].append(KL_disagreement(ensemble_softmax, members_softmax))
        score_dict[key] = np.array(score_dict[key])
    return score_dict


def compute_L2_logit_disagreement(data_dict_ensemble, data_dict_members, num_models):
    score_dict = {}
    for key, output_key in zip(["known", "unknown"], ["preds_k", "preds_u"]):
        score_dict[key] = []
        # for every sample: compute the sum of KLD(ensemble, model_i) over all i.
        for sample_id in range(len(data_dict_ensemble[output_key])):
            ensemble_logit = data_dict_ensemble[output_key][sample_id, :][None, :]
            members_logit = np.array(
                [
                    data_dict_members[model_id][output_key][sample_id, :]
                    for model_id in range(num_models)
                ]
            )
            score_dict[key].append(float(np.sum((ensemble_logit - members_logit) ** 2)))
        score_dict[key] = np.array(score_dict[key])
    return score_dict


def compute_cossim_logit_disagreement(
    data_dict_ensemble, data_dict_members, num_models, stats_fun
):
    """cosine similarity disagreement: Sum of the cosine similarity between ensemble logits and member logits"""
    score_dict = {}
    for key, output_key in zip(["known", "unknown"], ["preds_k", "preds_u"]):
        score_dict[key] = []
        # for every sample: compute the sum of KLD(ensemble, model_i) over all i.
        for sample_id in range(len(data_dict_ensemble[output_key])):
            ensemble_logit = data_dict_ensemble[output_key][sample_id, :][None, :]
            members_logit = np.array(
                [
                    data_dict_members[model_id][output_key][sample_id, :]
                    for model_id in range(num_models)
                ]
            )
            cosine_similarity_ensemble_to_members = cosine_similarity(
                ensemble_logit, members_logit
            )
            score_dict[key].append(
                float(stats_fun(cosine_similarity_ensemble_to_members))
            )
        score_dict[key] = np.array(score_dict[key])
    return score_dict


def compute_stats_of_member_score(
    data_dict_members, output_keys, num_models, stats_fun
):
    score_dict = {}
    for key, output_key in zip(["known", "unknown"], output_keys):
        score_per_member = np.array(
            [
                np.max(data_dict_members[model_id][output_key], axis=1)
                for model_id in range(num_models)
            ]
        )
        # print("score_per_member.shape", score_per_member.shape)
        score_dict[key] = stats_fun(score_per_member, axis=0)
        # print("score_dict[key].shape", score_dict[key].shape)
    return score_dict


def torch_max_values(x, dim):
    return torch.max(x, dim).values


def torch_std_neg(x, dim):
    return -torch.std(x, dim=dim)


def coeff_var(x, dim):
    return -torch.std(x, dim=dim) / torch.mean(x, dim=dim)


def combine_outputs_testdatasets(scores_dict, dataset_train, dataset_test_list, test_hops):
    scores_dict_combined = init_nested_dict()
    for score_key in scores_dict[dataset_train][dataset_test_list[0]]["ensemble"].keys():
        for subset_key in ["known", "unknown"]:
            scores_list = []
            
            for dataset_test, test_hop in zip(dataset_test_list, test_hops):
                scores_i = scores_dict[dataset_train][dataset_test]["ensemble"][score_key][subset_key]
                scores_list.append(scores_i)

            scores_dict_combined[score_key][subset_key] = np.concatenate(scores_list)
    
    # test_hops are the same for all scores. use the last score and len "unknown"
    hops_list = []
    for dataset_test, test_hop in zip(dataset_test_list, test_hops):
        hops_list.append(np.repeat(test_hop, len(scores_dict[dataset_train][dataset_test]["ensemble"][score_key]["unknown"])))
    test_hops_combined = np.concatenate(hops_list)   
            
    return scores_dict_combined, test_hops_combined


if __name__ == "__main__":

    args = parser.parse_args()

    save_dir = os.path.join(args.model_path, f"ensemble_models{args.num_models}")
    # add ensemble_id to save_dir
    if args.select_random_ids:
        save_dir = save_dir + f"_id{args.ensemble_id}"
    print("save_dir: ", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_test_list = [
        args.dataset_train.replace("1hop", f"{hop}hop") for hop in args.test_hops
    ]

    # get list of model_id paths
    if args.select_random_ids:
        np.random.seed(args.ensemble_id)
        args.model_ids = np.random.choice(range(args.num_models_pool), size=args.num_models, replace=False)
        print("model_ids: ", args.model_ids)
    else:
        args.model_ids = list(range(args.num_models))
    
    args.model_paths = get_model_paths(model_path=args.model_path, model_ids=args.model_ids)

    # ==================================================
    # load test metrics and compute statistics
    # ==================================================

    # load test metrics of individual models
    metrics_dict_member = load_member_test_metrics(
        model_paths=args.model_paths,
        dataset_train=args.dataset_train,
        dataset_test_list=dataset_test_list,
    )
    save_metrics_to_json(
        metrics_dict_member, filepath=os.path.join(save_dir, "metrics_members.json")
    )

    # get statistics of members' test metrics (min, max, mean, std)
    stats_fun_dict = dict(
        zip(["min", "max", "mean", "std"], [np.min, np.max, np.mean, np.std])
    )
    metrics_dict_member_stats = compute_member_test_metrics_stats(
        metrics_dict=metrics_dict_member,
        stats_fun_dict=stats_fun_dict,
        num_models=args.num_models,
        dataset_train=args.dataset_train,
        dataset_test_list=dataset_test_list,
    )
    save_metrics_to_json(
        metrics_dict_member_stats,
        filepath=os.path.join(save_dir, "metrics_member_stats.json"),
    )

    # ==================================================
    # load model_outputs of individual models
    # ==================================================
    data_dict_members = load_member_model_outputs(
        model_paths=args.model_paths,
        dataset_train=args.dataset_train,
        dataset_test_list=dataset_test_list,
    )
    save_model_outputs_to_npz(
        data_dict_members, filepath=os.path.join(save_dir, "model_outputs_members.npz")
    )

    # ==================================================
    # average ensemble model output (average of members)
    # ==================================================
    # average of: 'preds_k', 'preds_u', 'preds_k_probs', 'preds_u_probs', 'feat_k', 'feat_u'
    data_dict_ensemble = compute_ensemble_model_output(
        data_dict=data_dict_members,
        num_models=args.num_models,
        dataset_train=args.dataset_train,
        dataset_test_list=dataset_test_list,
    )
    save_model_outputs_to_npz(
        data_dict_ensemble,
        filepath=os.path.join(save_dir, "model_outputs_ensemble.npz"),
    )

    # ==================================================
    # compute OSR scores for known and unknown test data
    # ==================================================
    # init dictionary: scores_dict[args.dataset_train][dataset_test]["ensemble"][score_name] = {"known": [], "unknown": []}
    scores_dict = init_nested_dict()

    for dataset_test in dataset_test_list:
        print("computing scores for dataset_test: ", dataset_test)

        # ------- ENSEMBLE AVERAGE -------

        # max-ensemble-logit
        scores_dict[args.dataset_train][dataset_test]["ensemble"]["max-ensemble-logit"] = {
            "known": np.max(
                data_dict_ensemble[args.dataset_train][dataset_test]["ensemble"]["preds_k"],
                axis=1,
            ),
            "unknown": np.max(
                data_dict_ensemble[args.dataset_train][dataset_test]["ensemble"]["preds_u"],
                axis=1,
            ),
        }

        # max-ensemble-softmax
        scores_dict[args.dataset_train][dataset_test]["ensemble"]["max-ensemble-softmax"] = {
            "known": np.max(
                data_dict_ensemble[args.dataset_train][dataset_test]["ensemble"][
                    "preds_k_probs"
                ],
                axis=1,
            ),
            "unknown": np.max(
                data_dict_ensemble[args.dataset_train][dataset_test]["ensemble"][
                    "preds_u_probs"
                ],
                axis=1,
            ),
        }

        # negative entropy-ensemble-softmax (a high score means known)
        scores_dict[args.dataset_train][dataset_test]["ensemble"][
            "entropy-ensemble-softmax"
        ] = {
            "known": -entropy(
                data_dict_ensemble[args.dataset_train][dataset_test]["ensemble"][
                    "preds_k_probs"
                ],
                axis=1,
            ),
            "unknown": -entropy(
                data_dict_ensemble[args.dataset_train][dataset_test]["ensemble"][
                    "preds_u_probs"
                ],
                axis=1,
            ),
        }

        # ------- ENSEMBLE DISAGREEMENT (epistemic uncertainty) -------

        # KLD-softmax-disagree
        scores_dict[args.dataset_train][dataset_test]["ensemble"][
            "KLD-disagreement"
        ] = compute_KLD_disagreement(
            data_dict_ensemble=data_dict_ensemble[args.dataset_train][dataset_test][
                "ensemble"
            ],
            data_dict_members=data_dict_members[args.dataset_train][dataset_test],
            num_models=args.num_models,
        )

        # negative KLD: change the sign (a high score means known)
        for k in ["known", "unknown"]:
            scores_dict[args.dataset_train][dataset_test]["ensemble"]["KLD-disagreement"][
                k
            ] *= -1

        # negative L2-logit-disagree
        scores_dict[args.dataset_train][dataset_test]["ensemble"][
            "L2-logit-disagreement"
        ] = compute_L2_logit_disagreement(
            data_dict_ensemble=data_dict_ensemble[args.dataset_train][dataset_test][
                "ensemble"
            ],
            data_dict_members=data_dict_members[args.dataset_train][dataset_test],
            num_models=args.num_models,
        )
        # negative L2 disagreement: change the sign (a high score means known)
        for k in ["known", "unknown"]:
            scores_dict[args.dataset_train][dataset_test]["ensemble"][
                "L2-logit-disagreement"
            ][k] *= -1

        #  cosine similarity logit disagreement (mean and variance)
        scores_dict[args.dataset_train][dataset_test]["ensemble"][
            "cossim-mean-logit-disagreement"
        ] = compute_cossim_logit_disagreement(
            data_dict_ensemble=data_dict_ensemble[args.dataset_train][dataset_test][
                "ensemble"
            ],
            data_dict_members=data_dict_members[args.dataset_train][dataset_test],
            num_models=args.num_models,
            stats_fun=np.mean,
        )

        scores_dict[args.dataset_train][dataset_test]["ensemble"][
            "cossim-var-logit-disagreement"
        ] = compute_cossim_logit_disagreement(
            data_dict_ensemble=data_dict_ensemble[args.dataset_train][dataset_test][
                "ensemble"
            ],
            data_dict_members=data_dict_members[args.dataset_train][dataset_test],
            num_models=args.num_models,
            stats_fun=np.var,
        )

        # negative cosine similarity variance: change the sign (a high score means known)
        for k in ["known", "unknown"]:
            scores_dict[args.dataset_train][dataset_test]["ensemble"][
                "cossim-var-logit-disagreement"
            ][k] *= -1

        # ------- STATS OF MEMBER SCORES -------

        # max/min/mean/std-max-member-logit/softmax
        output_key_lookup = {
            "logit": ["preds_k", "preds_u"],
            "softmax": ["preds_k_probs", "preds_u_probs"],
        }

        for output_name, output_keys in output_key_lookup.items():
            for stats_name, stats_fun in stats_fun_dict.items():
                score_name = "{}-max-member-{}".format(stats_name, output_name)

                scores_dict[args.dataset_train][dataset_test]["ensemble"][
                    score_name
                ] = compute_stats_of_member_score(
                    data_dict_members=data_dict_members[args.dataset_train][dataset_test],
                    output_keys=output_keys,
                    num_models=args.num_models,
                    stats_fun=stats_fun,
                )

        # ------- KNN (cosine distance and L2 norm) -------
        batch_size = 1
        k_list = [1, 3, 5, 10, 20, 50, 100, 200]

        stats_fun_dict_knn = {
            "mean": torch.mean,
            "max": torch_max_values,
        }  

        dist_keys = ["L2-l2norm", "L2-unnorm"]  # alternative, but slow: "cos-unnorm"

        if len(dist_keys) > 0:

            dists_dict = {
                key: {"dist": key.split("-")[0], "feat_norm": key.split("-")[1]}
                for key in dist_keys
            }

            # load member train features and compute ensemble average
            print("loading member train_feats...")
            data_dict_members_train = load_member_model_outputs_train(
                model_paths=args.model_paths, dataset_train=args.dataset_train, output_keys=["feat_k"]
            )
            print("computing ensemble train_feats...")
            train_feats = compute_ensemble_model_output_train(
                data_dict=data_dict_members_train,
                num_models=args.num_models,
                dataset_train=args.dataset_train,
                output_keys=["feat_k"],
            )[args.dataset_train]["ensemble"]["feat_k"]
            # convert to tensor and move to gpu
            train_feats = torch.tensor(train_feats).cuda()
            print("train_feats.size()", train_feats.size())

            # compute distances and knn to train data for known and unknown test data
            for subset_key, feat_key in zip(["known", "unknown"], ["feat_k", "feat_u"]):
                for dist_key in dist_keys:
                    print(dist_key)
                    test_feats = torch.tensor(
                        data_dict_ensemble[args.dataset_train][dataset_test]["ensemble"][
                            feat_key
                        ]
                    ).cuda()

                    # loop through each test feature and append knn score
                    for feat in tqdm(test_feats.split(batch_size)):
                        dists = get_distance_matrix(
                            train_feats,
                            feat,
                            dist_key=dists_dict[dist_key]["dist"],
                            feat_norm=dists_dict[dist_key]["feat_norm"],
                        )

                        # get knn for largest k
                        knn = dists.topk(
                            k=max(k_list), largest=False, sorted=True, dim=1
                        )  # knn.values, knn.indices
                        # print(f"k: {max(k_list)}, knn_dists.size: {knn.values.size()}")
                        for k in k_list:
                            # select k from knn
                            knn_dists = knn.values[:, :k]
                            # print(f"k: {k}, knn_dists.size: {knn_dists.size()}")
                            for stats_key in stats_fun_dict_knn.keys():
                                score_key = f"knn-{dist_key}-k{k}-{stats_key}"
                                # init empty list if the first time:
                                if (
                                    subset_key
                                    not in scores_dict[args.dataset_train][dataset_test][
                                        "ensemble"
                                    ][score_key]
                                ):
                                    scores_dict[args.dataset_train][dataset_test]["ensemble"][
                                        score_key
                                    ][subset_key] = []
                                # append the knn score
                                # Note: for consitency we use the negative distance as a score, such that high scores correspond to familiar categories
                                scores_dict[args.dataset_train][dataset_test]["ensemble"][
                                    score_key
                                ][subset_key].append(
                                    -stats_fun_dict_knn[stats_key](knn_dists, dim=1).item()
                                )

        # convert list to np array
        for score_key in scores_dict[args.dataset_train][dataset_test]["ensemble"].keys():
            for subset_key in ["known", "unknown"]:
                scores_dict[args.dataset_train][dataset_test]["ensemble"][score_key][
                    subset_key
                ] = np.array(
                    scores_dict[args.dataset_train][dataset_test]["ensemble"][score_key][
                        subset_key
                    ]
                )

        # save scores_dict to npz
        save_model_outputs_to_npz(
            scores_dict, filepath=os.path.join(save_dir, "ensemble_scores_dict.npz")
        )

    # ==================================================
    # evaluate OSR performance of ensemble statistics
    # ==================================================
    ensemble_metrics_dict = init_nested_dict()

    for dataset_test in dataset_test_list:
        print("===========================================")
        print("evaluating osr metrics for dataset_test: ", dataset_test)
        score_keys = list(scores_dict[args.dataset_train][dataset_test]["ensemble"].keys())

        # get number of samples in known and unknown test set
        num_samples_k = len(
            data_dict_ensemble[args.dataset_train][dataset_test]["ensemble"]["preds_k"]
        )
        num_samples_u = len(
            data_dict_ensemble[args.dataset_train][dataset_test]["ensemble"]["preds_u"]
        )

        for score_key in score_keys:
            # print("score: ", score_key)
            score_k = scores_dict[args.dataset_train][dataset_test]["ensemble"][score_key][
                "known"
            ]
            score_u = scores_dict[args.dataset_train][dataset_test]["ensemble"][score_key][
                "unknown"
            ]
            assert len(score_k) == num_samples_k
            assert len(score_u) == num_samples_u

            labels = data_dict_ensemble[args.dataset_train][dataset_test]["ensemble"]["labels"]
            labels_u = data_dict_ensemble[args.dataset_train][dataset_test]["ensemble"]["labels_u"]
            print("labels_u.shape", labels_u.shape)
            print("labels.shape", labels.shape)
            probs_k = data_dict_ensemble[args.dataset_train][dataset_test]["ensemble"]["preds_k_probs"]

            ood_metrics = evaluation.metric_ood(score_k, score_u, labels=labels, probs_k=probs_k, stypes=[score_key],
            )[score_key]

            # sample level ranking metrics (novel vs. known)
            ranking_metrics = evaluation.metrics_ranking(score_k, score_u, topk_list=[10, 100, 1000])
            ood_metrics.update(ranking_metrics)

            ensemble_metrics_dict[args.dataset_train][dataset_test]["ensemble"][
                score_key
            ] = ood_metrics
    
    
    # ==================================================
    # save osr test metrics for ensemble scores
    # ==================================================
    save_metrics_to_json(
        ensemble_metrics_dict,
        filepath=os.path.join(save_dir, "ensemble_metrics_dict.json"),
    )

    # save args to json
    save_metrics_to_json(vars(args), filepath=os.path.join(save_dir, "args.json"))

