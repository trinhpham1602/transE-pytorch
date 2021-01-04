from absl import app
from absl import flags
import data
import TransE as TransE_definition
import os
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
import glob
FLAGS = flags.FLAGS
flags.DEFINE_float("lr", default=0.0001, help="Learning rate value.")
flags.DEFINE_integer("seed", default=1234, help="Seed value.")
flags.DEFINE_integer("batch_size", default=128, help="Maximum batch size.")
flags.DEFINE_integer("validation_batch_size", default=64,
                     help="Maximum batch size during model validation.")
flags.DEFINE_integer("emb_dim", default=100,
                     help="Length of entity/relation vector.")
flags.DEFINE_float("margin", default=1.0,
                   help="Margin value in margin-based ranking loss.")
flags.DEFINE_integer(
    "norm", default=1, help="Norm used for calculating dissimilarity metric (usually 1 or 2).")
flags.DEFINE_integer("epochs", default=2000,
                     help="Number of training epochs.")
flags.DEFINE_string("dataset_path", default="./data/FB15k-237",
                    help="Path to dataset.")
flags.DEFINE_bool("use_gpu", default=True, help="Flag enabling gpu usage.")
# flags.DEFINE_integer("validation_freq", default=10,
#                      help="Validate model every X epochs.")
# flags.DEFINE_string("checkpoint_path", default="",
#                     help="Path to model checkpoint (by default train from scratch).")
# flags.DEFINE_string("tensorboard_log_dir", default="./runs",
#                     help="Path for tensorboard log directory.")

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
    train_file = open(path + "/" + "train.txt", "rb")
    valid_file = open(path + "/" + "valid.txt", "rb")
    test_file = open(path + "/" + "test.txt", "rb")
    with open(path + "/" + "all_triples.txt", "wb") as outfile:
        outfile.write(train_file.read())
        outfile.write(valid_file.read())
        outfile.write(test_file.read())

    train_path = os.path.join(path, "all_triples.txt")

    entity2id, relation2id = data.create_mappings(train_path)

    batch_size = FLAGS.batch_size
    emb_dim = FLAGS.emb_dim
    margin = FLAGS.margin
    norm = FLAGS.norm
    learning_rate = FLAGS.lr
    epochs = FLAGS.epochs
    device = torch.device('cuda') if FLAGS.use_gpu else torch.device('cpu')

    train_set = data.KGDataset(train_path, entity2id, relation2id)
    N_triples = train_set.__len__()
    train_generator = torch_data.DataLoader(train_set, batch_size=batch_size)

    model = TransE_definition.TransE(entity_count=len(entity2id), relation_count=len(relation2id), dim=emb_dim,
                                     margin=margin,
                                     device=device, norm=norm)  # type: torch.nn.Module
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    start_epoch_id = 1
    neg_blocks = []
    pos_blocks = []
    print("training TransE with %d epochs!", epochs)
    for epoch in range(start_epoch_id, epochs + 1):
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
        print("Finished the epoch: ", epoch)
    # take k% lowest h + r - t
    print("---------------------------------------------")
    print("Start the training NoiAwareGANs")
    k = 0.7
    N = 100
    for time in range(N):
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
        k_percent_lowest = k_percent_lowest.to(device)
        # define GANs
        epochs4GANs = 1000
        D, G = GANs.run(k_percent_lowest, emb_dim,
                        learning_rate, batch_size, epochs4GANs)
        # train noiAwareKGE
        model = noiAware_difinition.NoiAwareKGE(
            model.entities_emb, model.relations_emb, emb_dim, device=device)
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.LogSigmoid()
        for _ in range(start_epoch_id, epochs + 1):
            model.train()
            for inx in range(len(train_generator)):
                optimizer.zero_grad()
                loss = model(pos_blocks[inx], neg_blocks[inx], D, G)
                loss = criterion(loss)
                loss.mean().backward()
                optimizer.step()
        print("Finished interator: ", time + 1)
    end = perf_counter()
    print("The NoiAwareGAN is trained")
    print("total time pretrain and train NoiAwareGANs is ", end - start)
    print("---------------------------------------------")
    entities_emb = model.entities_emb.weight.data.cpu().numpy()
    relations_emb = model.relations_emb.weight.data.cpu().numpy()
    np.savetxt("./output/entities_emb.txt", entities_emb)
    np.savetxt("./output/relations_emb.txt", relations_emb)
    print("Done!")


if __name__ == '__main__':
    app.run(main)
