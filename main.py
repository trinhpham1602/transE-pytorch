from absl import app
from absl import flags
import data
import metric
import model as model_definition
import os
import storage
import torch
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils import tensorboard
from typing import Tuple
import numpy as np
import GANs
import NoiAwareKGE as NoiGANs_definition

FLAGS = flags.FLAGS
flags.DEFINE_float("lr", default=0.01, help="Learning rate value.")
flags.DEFINE_integer("seed", default=1234, help="Seed value.")
flags.DEFINE_integer("batch_size", default=128, help="Maximum batch size.")
flags.DEFINE_integer("validation_batch_size", default=64,
                     help="Maximum batch size during model validation.")
flags.DEFINE_integer("vector_length", default=50,
                     help="Length of entity/relation vector.")
flags.DEFINE_float("margin", default=1.0,
                   help="Margin value in margin-based ranking loss.")
flags.DEFINE_integer(
    "norm", default=1, help="Norm used for calculating dissimilarity metric (usually 1 or 2).")
flags.DEFINE_integer("epochs", default=1, help="Number of training epochs.")
flags.DEFINE_string("dataset_path", default="./data/WN18RR",
                    help="Path to dataset.")
flags.DEFINE_bool("use_gpu", default=True, help="Flag enabling gpu usage.")
flags.DEFINE_integer("validation_freq", default=10,
                     help="Validate model every X epochs.")
flags.DEFINE_string("checkpoint_path", default="",
                    help="Path to model checkpoint (by default train from scratch).")
flags.DEFINE_string("tensorboard_log_dir", default="./runs",
                    help="Path for tensorboard log directory.")

HITS_AT_1_SCORE = float
HITS_AT_3_SCORE = float
HITS_AT_10_SCORE = float
MRR_SCORE = float
METRICS = Tuple[HITS_AT_1_SCORE, HITS_AT_3_SCORE, HITS_AT_10_SCORE, MRR_SCORE]


def main(_):
    torch.random.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    path = FLAGS.dataset_path
    train_path = os.path.join(path, "train.txt")
    #validation_path = os.path.join(path, "valid.txt")
    #test_path = os.path.join(path, "test.txt")

    entity2id, relation2id = data.create_mappings(train_path)

    batch_size = FLAGS.batch_size
    vector_length = FLAGS.vector_length
    margin = FLAGS.margin
    norm = FLAGS.norm
    learning_rate = FLAGS.lr
    epochs = FLAGS.epochs
    device = torch.device('cuda') if FLAGS.use_gpu else torch.device('cpu')

    train_set = data.FB15KDataset(train_path, entity2id, relation2id)
    train_generator = torch_data.DataLoader(train_set, batch_size=batch_size)

    # save id head-relation-tail

    #####################################################################
    model = model_definition.TransE(entity_count=len(entity2id), relation_count=len(relation2id), dim=vector_length,
                                    margin=margin,
                                    device=device, norm=norm)  # type: torch.nn.Module
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    start_epoch_id = 1
    print("Pretrain with TransE!")
    print("_____________________")
    print("training TransE")

    # Training loop
    entities_id_order = {}
    relations_id_order = {}

    for _ in range(start_epoch_id, epochs + 1):
        model.train()

        for local_heads, local_relations, local_tails in train_generator:
            local_heads, local_relations, local_tails = (local_heads.to(device), local_relations.to(device),
                                                         local_tails.to(device))
            entities = torch.cat(
                (local_heads, local_tails), 0).view(-1).numpy()
            relations = local_relations.view(-1).numpy()
######################################################
            for inx in range(len(entities)):
                if entities[inx] not in entities_id_order.keys():
                    entities_id_order[entities[inx]] = len(
                        entities_id_order)

            for inx in range(len(relations)):
                if relations[inx] not in relations_id_order.keys():
                    relations_id_order[relations[inx]] = len(
                        relations_id_order)
######################################################
            positive_triples = torch.stack(
                (local_heads, local_relations, local_tails), dim=1)
            # Preparing negatives.
            # Generate binary tensor to replace either head or tail. 1 means replace head, 0 means replace tail.
            head_or_tail = torch.randint(
                high=2, size=local_heads.size(), device=device)
            random_entities = torch.randint(
                high=len(entity2id), size=local_heads.size(), device=device)
            broken_heads = torch.where(
                head_or_tail == 1, random_entities, local_heads)
            broken_tails = torch.where(
                head_or_tail == 0, random_entities, local_tails)
            negative_triples = torch.stack(
                (broken_heads, local_relations, broken_tails), dim=1)

            optimizer.zero_grad()

            loss, _, _ = model(positive_triples, negative_triples)
            loss.mean().backward()

            optimizer.step()
    # mapping embedding vector with id order
    ##########################################################################
    positive_emb_matrix = np.empty(
        (len(train_set), 3, vector_length), dtype=float)
    dtype = [("distance", float), ("order", int)]
    distances_inx = []
    target_hrt_order = []
    for inx in range(len(train_set.data)):
        (h, r, t) = train_set.__getitem__(inx)
        target_hrt_order.append([h, r, t])
        head_order = entities_id_order[h]
        tail_order = entities_id_order[t]
        relation_order = relations_id_order[r]
        positive_emb_matrix[inx] = [model.entities_emb.weight.data.numpy()[head_order], model.relations_emb.weight.data.numpy()[
            relation_order], model.entities_emb.weight.data.numpy()[tail_order]]
        distances_inx.append((np.linalg.norm((model.entities_emb.weight.data.numpy()[head_order] + model.relations_emb.weight.data.numpy()[
            relation_order] - model.entities_emb.weight.data.numpy()[tail_order]), ord=1), inx))

    ##########################################################################
    print("Training TransE has finished!!!")
    # take 10% norm_1(h+r-t) smallest
    print("_______________________________")
    print("Start the training NoiGANs")
    # k persent is choosen 10%
    k = 0.1
    N = 1
    for i in range(0, N):
        temp = np.array(distances_inx, dtype=dtype)
        temp = np.sort(temp, order="distance")
        k_percent_lowest = temp[:int(len(distances_inx)*k), ]
        target_k_percent = positive_emb_matrix[[
            val[1] for val in k_percent_lowest]]
        # sent k_percent to GANs model
        hrt_vecs = torch.Tensor([target_k_percent[i][0] + target_k_percent[i][1] +
                                 target_k_percent[i][2] for i in range(0, len(target_k_percent))]).float()
        # training GANs with 200 epoches
        lr4GANs = 0.01
        batch_size_in_k_percent = 256
        epoches4GAns = 1
        D, G = GANs.run(hrt_vecs, vector_length, lr4GANs,
                        batch_size_in_k_percent, epoches4GAns)

        all_hrt = torch.from_numpy(
            positive_emb_matrix).float()
        concat_all_hrt = torch.reshape(
            all_hrt, ((len(all_hrt), 1, 3*vector_length))).float()
        generated_neg_triplets = G.forward(concat_all_hrt)
        generated_neg_triplets = torch.reshape(
            generated_neg_triplets, ((len(all_hrt), 3, vector_length)))
        all_hrt_computed = torch.sum(all_hrt, dim=1).float()
        C_scores = D.forward(
            all_hrt_computed)
        C_scores_4_all_hrt = torch.ones_like(concat_all_hrt)
        for i in range(len(concat_all_hrt)):
            C_scores_4_all_hrt[i] = C_scores[i]*concat_all_hrt[i]
        C_scores_4_all_hrt = torch.reshape(
            C_scores_4_all_hrt, ((len(all_hrt), 3, vector_length)))

        # create NoiGANs
        target_hrt_order = torch.tensor(target_hrt_order)
        noiGANs = NoiGANs_definition.NoiAwareKGE(
            n_samples=len(C_scores_4_all_hrt), device=device, emb_dim=vector_length, norm=norm, margin=margin)
        noiGANs = noiGANs.to(device)
        optimizer = optim.SGD(noiGANs.parameters(), lr=learning_rate)
        loader = torch_data.DataLoader(
            range(len(C_scores_4_all_hrt)), batch_size)
        for _ in range(start_epoch_id, epochs + 1):
            noiGANs.train()
            for inx in loader:
                pos_samples = C_scores_4_all_hrt[inx.numpy()]
                pos_samples = pos_samples.to(device)
                neg_samples = generated_neg_triplets[inx.numpy()]
                neg_samples = neg_samples.to(device)

                optimizer.zero_grad()
                print("a")
                loss = noiGANs(pos_samples, neg_samples, inx)
                print("aa")
                loss.mean().backward()
                print("aaa")
                optimizer.step()

    print("Training the NoiGANs has finished!!!")


if __name__ == '__main__':
    app.run(main)
