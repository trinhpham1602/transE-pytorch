import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class Discriminator(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, emb_dim_concat):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(emb_dim_concat, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim_concat),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.gen(x)


def run(hrt_vecs, emb_dim, lr, batch_size, epoches):
    disc = Discriminator(emb_dim).to(device)
    gen = Generator(3*emb_dim).to(device)

    loader = DataLoader(hrt_vecs, batch_size=batch_size, shuffle=True)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for _ in range(epoches):
        for hrt_vec in loader:

            # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            noise = torch.randn(batch_size, 3*emb_dim).to(
                device)  # random batch_size dòng và emb_dim cột
            fake_hrt = gen(noise)
            fake_hrt = fake_hrt.detach().numpy().reshape((len(fake_hrt), 3, emb_dim))
            fake_hrt = np.array([fake_hrt[i][0] + fake_hrt[i][1] + fake_hrt[i][2]
                                 for i in range(0, len(fake_hrt))])
            fake_hrt = torch.from_numpy(fake_hrt).float()
            disc_real = disc(hrt_vec.float()).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake_hrt).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # where the second option of maximizing doesn't suffer from
            # saturating gradients
            output = disc(fake_hrt).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()
    return disc, gen
