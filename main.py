from absl import app
from absl import flags
import data
import metric
import model as model_definition
import os
import storage
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils import data as torch_data
from torch.utils import tensorboard
from typing import Tuple
import numpy as np
import GANs
import noiAwareKGE as noiAware_difinition
from time import perf_counter
FLAGS = flags.FLAGS
flags.DEFINE_float("lr", default=0.01, help="Learning rate value.")
flags.DEFINE_integer("seed", default=1234, help="Seed value.")
flags.DEFINE_integer("batch_size", default=128, help="Maximum batch size.")
flags.DEFINE_integer("validation_batch_size", default=64,
                     help="Maximum batch size during model validation.")
flags.DEFINE_integer("emb_dim", default=50,
                     help="Length of entity/relation vector.")
flags.DEFINE_float("margin", default=1.0,
                   help="Margin value in margin-based ranking loss.")
flags.DEFINE_integer(
    "norm", default=1, help="Norm used for calculating dissimilarity metric (usually 1 or 2).")
flags.DEFINE_integer("epoches", default=1, help="Number of training epoches.")
flags.DEFINE_string("dataset_path", default="./data/WN18RR",
                    help="Path to dataset.")
flags.DEFINE_bool("use_gpu", default=True, help="Flag enabling gpu usage.")
flags.DEFINE_integer("validation_freq", default=10,
                     help="Validate model every X epoches.")
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
    start = perf_counter()
    torch.random.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    path = FLAGS.dataset_path
    train_path = os.path.join(path, "train.txt")
    #validation_path = os.path.join(path, "valid.txt")
    #test_path = os.path.join(path, "test.txt")

    entity2id, relation2id = data.create_mappings(train_path)

    batch_size = FLAGS.batch_size
    emb_dim = FLAGS.emb_dim
    margin = FLAGS.margin
    norm = FLAGS.norm
    learning_rate = FLAGS.lr
    epoches = FLAGS.epoches
    device = torch.device('cuda') if FLAGS.use_gpu else torch.device('cpu')

    train_set = data.KGDataset(train_path, entity2id, relation2id)
    N_triples = train_set.__len__()
    train_generator = torch_data.DataLoader(train_set, batch_size=batch_size)

    model = model_definition.TransE(entity_count=len(entity2id), relation_count=len(relation2id), dim=emb_dim,
                                    margin=margin,
                                    device=device, norm=norm)  # type: torch.nn.Module
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    start_epoch_id = 1
    neg_blocks = []
    pos_blocks = []
    for _ in range(start_epoch_id, epoches + 1):
        model.train()

        for local_heads, local_relations, local_tails in train_generator:
            local_heads, local_relations, local_tails = (local_heads.to(device), local_relations.to(device),
                                                         local_tails.to(device))

            positive_triples = torch.stack(
                (local_heads, local_relations, local_tails), dim=1)
            pos_blocks.append(positive_triples)
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
            neg_blocks.append(negative_triples)

            optimizer.zero_grad()

            loss, _, _ = model(positive_triples, negative_triples)
            loss.mean().backward()

            optimizer.step()
    # take k% lowest h + r - t
    k = 0.4
    N = 1
    for i in range(N):
        entities_emb = model.entities_emb.weight.data
        relations_emb = model.relations_emb.weight.data
        hrt_embs = torch.zeros((N_triples, 3, emb_dim), dtype=float)
        norm_order = []
        #triples_id = []
        # in train_set, lines is random and diference with origin data set
        for i in range(N_triples):
            (h_id, r_id, t_id) = train_set.__getitem__(i)
            h_emb = entities_emb[h_id]
            r_emb = relations_emb[r_id]
            t_emb = entities_emb[t_id]
            norm = torch.norm(h_emb+r_emb-t_emb, p=1)
            norm_order.append((norm.item(), i))
            #triples_id.append([h_id, r_id, t_id])
            hrt_embs[i][0] = h_emb
            hrt_embs[i][1] = r_emb
            hrt_embs[i][2] = t_emb

        dtype = [("norm", float), ("order", int)]
        norm_order = np.array(norm_order, dtype=dtype)
        norm_order = np.sort(norm_order, order="norm")
        k_percent_lowest = torch.zeros((int(k*N_triples), 3, emb_dim))
        for i in range(int(k*N_triples)):
            k_percent_lowest[i] = hrt_embs[norm_order[i][1]]

        # define GANs
        D, G = GANs.run(k_percent_lowest, emb_dim,
                        learning_rate, batch_size, epoches)
        # train noiAwareKGE
        model = noiAware_difinition.NoiAwareKGE(
            model.entities_emb, model.relations_emb, emb_dim, device=device)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.LogSigmoid()
        for _ in range(start_epoch_id, epoches + 1):
            model.train()
            for inx in range(len(train_generator)):
                optimizer.zero_grad()
                loss = model(pos_blocks[inx], neg_blocks[inx], D, G)
                loss = criterion(loss)
                loss.mean().backward()
                optimizer.step()
    end = perf_counter()
    print(end - start)


if __name__ == '__main__':
    app.run(main)
