import os
import numpy as np
import json

from config import inat_2021_root, parent_dir

from inat2021osr import inat21_supercat_dict
from inat2021 import (
    load_taxonomy,
    get_num_categories_per_taxonomic_rank,
    ids_to_tax_rank,
    get_num_samples_per_tax_id,
    load_tax_id_name_dicts,
    tax_ids_to_species_ids,
    make_sub_taxonomy_from_species_ids,
    compute_hop_distances,
    compute_min_hop_distances,
    tax_id_to_supertax_id,
)


if __name__ == "__main__":
    """
    For every super-category in the dictionary inat21_supercat_dict, 
    create a subset of closed-set ids and corresponding 1-7 hop open-set splits.
    """

    # set random seed
    seed = 1

    drop_factor = 0.2  # at every tax_level, keep 20% of tax_ids with the highest number of samples in the training set
    ood_factor = 0.2  # set 20% of tax_id candidates as open-set

    train_file = os.path.join(inat_2021_root, "train.json")
    val_file = os.path.join(inat_2021_root, "val.json")

    out_dir = os.path.join(inat_2021_root, "inat21_osr_splits")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load train data
    with open(train_file) as f:
        ann_data_train = json.load(f)
    # load val data
    with open(val_file) as f:
        ann_data_val = json.load(f)

    classes = [aa["category_id"] for aa in ann_data_train["annotations"]]

    # inat_2021   10000, 4884,    1103,     273,     51,      13,       3
    # inat_2018    8142, 4412,    1120,     273,     57,      25,       6
    tax_levels = ["id", "genus", "family", "order", "class", "phylum", "kingdom"]
    hops = range(1, 8)
    tax_levels_to_hops = dict(zip(tax_levels, hops))
    hops_to_tax_levels = dict(zip(hops, tax_levels))

    # load taxonomy
    taxonomy, classes_taxonomic = load_taxonomy(ann_data_train, tax_levels, classes)

    num_taxonomic_rank = get_num_categories_per_taxonomic_rank(taxonomy)
    print(num_taxonomic_rank)

    for supercat in inat21_supercat_dict.keys():
        np.random.seed(seed=seed)
        print("#############################################")
        print("supercat: ", supercat)
        print("#############################################")
        tax_level = inat21_supercat_dict[supercat]["tax_level"]
        key = inat21_supercat_dict[supercat]["key"]

        # create a sub taxonomy for the super category
        # load the mapping from rank_categories to rank_names
        rank_categories_to_names, rank_names_to_categories = load_tax_id_name_dicts(
            ann_data_train, tax_levels=[tax_level]
        )

        # get all species ids for supercat
        species_ids_sub = [
            k
            for k, v in taxonomy[tax_level].items()
            if v == rank_names_to_categories[tax_level][key]
        ]
        print(f"Num species in {supercat}: {len(species_ids_sub)}")

        taxonomy_sub = make_sub_taxonomy_from_species_ids(taxonomy, species_ids_sub)
        num_taxonomic_rank = get_num_categories_per_taxonomic_rank(taxonomy_sub)
        print(num_taxonomic_rank)

        # get num_samples per category for each tax level
        taxa_num_samples_per_tax_id = {}
        for tax_level in tax_levels:
            taxa_num_samples_per_tax_id[tax_level] = get_num_samples_per_tax_id(
                ann_data=ann_data_train, tax_level=tax_level, taxonomy=taxonomy_sub
            )
            # print(tax_level, taxa_num_samples_per_tax_id[tax_level])

        # ----------------------

        ood_tax_ids_per_tax_level = {}
        train_tax_ids_per_tax_level = {}

        ood_species_ids_per_tax_level = {}
        train_species_ids_per_tax_level = {}

        # select OOD categories per tax level
        # tax_levels = ['id', 'genus', 'family', 'order'][::-1]

        for i, tax_level in enumerate(tax_levels[::-1]):
            print("**************")
            print(tax_level)

            # sort dict by num_samples
            taxa_num_samples_per_tax_id[tax_level] = {
                k: v
                for k, v in sorted(
                    taxa_num_samples_per_tax_id[tax_level].items(),
                    key=lambda item: item[1],
                )
            }

            # set all ood candidates
            tax_id_candidates = list(taxa_num_samples_per_tax_id[tax_level].keys())
            num_samples_per_tax_id_candidates = list(
                taxa_num_samples_per_tax_id[tax_level].values()
            )
            print("tax_id_candidates all: ", len(tax_id_candidates))

            # drop tax_ids that contain the highest number of samples
            tax_id_candidates = tax_id_candidates[
                : -int(drop_factor * len(tax_id_candidates))
            ]
            num_samples_per_tax_id_candidates = num_samples_per_tax_id_candidates[
                : -int(drop_factor * len(tax_id_candidates))
            ]
            print("tax_id_candidates with low num samples: ", len(tax_id_candidates))

            # drop higher-level taxon ood ids
            if i > 0:
                # ----------------------
                # get species ids already in ood
                ood_species_ids = list(
                    np.concatenate(list(ood_species_ids_per_tax_level.values())).flat
                )
                # print(ood_species_ids)
                print("len(ood_species_ids)", len(ood_species_ids))
                ood_tax_ids = set(
                    ids_to_tax_rank(ood_species_ids, tax_level, taxonomy_sub)
                )
                print("len(ood_tax_ids)", len(ood_tax_ids))
                tax_id_candidates = list(set(tax_id_candidates) - set(ood_tax_ids))

                # ----------------------
                # drop one candidiate per super category. This makes sure that every ood candidate at hop-x has one sibling in the training set
                supertax_level = tax_levels[::-1][i - 1]
                print("supertax_level: ", supertax_level)
                supertax_id_candidates = tax_id_to_supertax_id(
                    tax_ids=tax_id_candidates,
                    tax_level_source=tax_level,
                    tax_level_target=supertax_level,
                    taxonomy=taxonomy_sub,
                )
                supertax_unique, supertax_counts = np.unique(
                    supertax_id_candidates, return_counts=True
                )
                # print(supertax_unique)
                # print(supertax_counts)
                # print(dict(zip(supertax_unique, supertax_counts)))

                tax_ids_to_drop = []
                for super_taxid in supertax_unique:
                    tax_ids_siblings = np.array(tax_id_candidates)[
                        supertax_id_candidates == super_taxid
                    ]
                    # select a random sibling and remove from candiates
                    tax_ids_to_drop.append(
                        np.random.choice(tax_ids_siblings, size=1).item()
                    )
                tax_id_candidates = list(set(tax_id_candidates) - set(tax_ids_to_drop))

            else:
                ood_tax_ids = set([])

            # select random open-set tax_ids
            if int(len(taxa_num_samples_per_tax_id[tax_level]) * ood_factor) < len(
                tax_id_candidates
            ):
                ood_size = int(len(taxa_num_samples_per_tax_id[tax_level]) * ood_factor)
            else:
                ood_size = int(len(tax_id_candidates) * 0.5)

            ood_tax_ids_per_tax_level[tax_level] = list(
                np.random.choice(
                    tax_id_candidates,
                    size=ood_size,
                    replace=False,
                    p=None,
                )
            )
            print(
                "ood_tax_ids_per_tax_level[tax_level]: ",
                len(ood_tax_ids_per_tax_level[tax_level]),
                type(ood_tax_ids_per_tax_level[tax_level]),
            )

            # get corresponding open-set species ids
            ood_species_ids_per_tax_level[tax_level] = tax_ids_to_species_ids(
                tax_ids=ood_tax_ids_per_tax_level[tax_level],
                tax_level=tax_level,
                taxonomy=taxonomy_sub,
            )
            print(
                "ood_species_ids_per_tax_level[tax_level]: ",
                len(ood_species_ids_per_tax_level[tax_level]),
            )

            # get remaining training tax_ids
            train_tax_ids_per_tax_level[tax_level] = list(
                set(taxa_num_samples_per_tax_id[tax_level].keys())
                - set(ood_tax_ids_per_tax_level[tax_level])
                - set(ood_tax_ids)
            )

        ood_species_ids = set(
            np.concatenate(list(ood_species_ids_per_tax_level.values())).flat
        )
        train_species_ids = tax_ids_to_species_ids(
            tax_ids=train_tax_ids_per_tax_level["id"],
            tax_level="id",
            taxonomy=taxonomy_sub,
        )
        print("============")
        print("len(ood_species_ids): ", len(ood_species_ids))
        print("len(train_species_ids): ", len(train_species_ids))
        print(
            "len(ood_species_ids) + len(train_species_ids)",
            len(ood_species_ids) + len(train_species_ids),
        )

        print(
            "train and ood isdisjoint:",
            set(train_species_ids).isdisjoint(ood_species_ids),
        )

        # -------------------
        # Get open-set splits for hops outside the train domain

        # get all remaining OSR species ids
        all_species_ids = taxonomy["id"].keys()
        print("len(all_species_ids)", len(all_species_ids))

        remaining_ood_species_ids = list(
            set(all_species_ids) - set(train_species_ids) - set(ood_species_ids)
        )
        print("len(remaining_ood_species_ids)", len(remaining_ood_species_ids))

        assert set(train_species_ids).isdisjoint(set(remaining_ood_species_ids))
        assert set(ood_species_ids).isdisjoint(set(remaining_ood_species_ids))
        print(
            "check sum: ",
            len(train_species_ids)
            + len(ood_species_ids)
            + len(remaining_ood_species_ids),
        )

        # compute hop distances between remaining ood ids and training ids
        dists = compute_hop_distances(
            remaining_ood_species_ids, train_species_ids, taxonomy=taxonomy
        )
        minimal_hop_dist = np.min(dists, axis=1)
        # print(np.unique(dists))
        print(minimal_hop_dist.shape)
        print(np.unique(minimal_hop_dist))
        print(minimal_hop_dist)

        # select ood ids for specific hop distance
        remaining_ood_species_ids = np.array(remaining_ood_species_ids)

        max_hop_dist_in_domain = int(
            np.max(
                np.min(
                    compute_hop_distances(
                        ood_species_ids, train_species_ids, taxonomy=taxonomy
                    ),
                    axis=1,
                )
            )
        )
        print("max_hop_dist_in_domain", max_hop_dist_in_domain)
        for hop in range(max_hop_dist_in_domain + 1, len(tax_levels) + 1):
            tax_level = hops_to_tax_levels[hop]
            ood_species_ids_per_tax_level[tax_level] = list(
                remaining_ood_species_ids[minimal_hop_dist == hop]
            )
            print(
                "number of ids in hop {}: {}".format(
                    hop, len(ood_species_ids_per_tax_level[tax_level])
                )
            )

        # -------------------
        # save json for hop 5 to 7
        print("================")
        print("saving to json...")
        for hop in range(1, len(tax_levels) + 1):
            tax_level = hops_to_tax_levels[hop]
            dataset_name = f"inat21-osr-{supercat}-id-{hop}hop"
            print(dataset_name)

            split_dict = {
                "train_categories": [int(v) for v in sorted(list(train_species_ids))],
                "test_categories": [
                    int(v)
                    for v in sorted(list(ood_species_ids_per_tax_level[tax_level]))
                ],
            }
            assert set(split_dict["train_categories"]).isdisjoint(
                set(split_dict["test_categories"])
            )

            # check min hop distances
            min_hop_dists = compute_min_hop_distances(
                split_dict["test_categories"], split_dict["train_categories"], taxonomy
            )
            print("unique min_hop_dists: ", np.unique(min_hop_dists))
            print(
                f"num train_ids: {len(split_dict['train_categories'])}, osr_ids: {len(split_dict['test_categories'])}"
            )
            with open(os.path.join(out_dir, dataset_name + ".json"), "w") as f:
                json.dump(split_dict, f)
