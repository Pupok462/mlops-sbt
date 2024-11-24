import torch
import lightning.pytorch as pl
import omegaconf


class GAN(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.generator = Generator(conf)
        self.discriminator = Discriminator(conf)

        self.loss_fn = torch.nn.BCELoss()

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        real_images, _ = batch
        batch_size = real_images.size(0)
        device = self.device

        optimizer_d, optimizer_g = self.optimizers()

        optimizer_d.zero_grad()

        real_labels = torch.ones(batch_size, 1, device=device)
        outputs = self.discriminator(real_images)
        real_loss = self.loss_fn(outputs, real_labels)

        z = torch.randn(batch_size, self.conf['model']['generator']['latent_size'], device=device)
        fake_images = self.generator(z)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        outputs = self.discriminator(fake_images.detach())
        fake_loss = self.loss_fn(outputs, fake_labels)

        d_loss = real_loss + fake_loss
        self.log('discriminator_loss', d_loss, on_step=True, on_epoch=True)

        self.manual_backward(d_loss)
        optimizer_d.step()

        optimizer_g.zero_grad()

        z = torch.randn(batch_size, self.conf['model']['generator']['latent_size'], device=device)
        fake_images = self.generator(z)
        outputs = self.discriminator(fake_images)
        real_labels = torch.ones(batch_size, 1, device=device)
        g_loss = self.loss_fn(outputs, real_labels)
        self.log('generator_loss', g_loss, on_step=True, on_epoch=True)

        self.manual_backward(g_loss)
        optimizer_g.step()

    def validation_step(self, batch, batch_idx):
        real_images, _ = batch
        batch_size = real_images.size(0)
        device = self.device

        # Оценка потерь дискриминатора
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        outputs = self.discriminator(real_images)
        real_loss = self.loss_fn(outputs, real_labels)

        z = torch.randn(batch_size, self.conf['model']['generator']['latent_size'], device=device)
        fake_images = self.generator(z)
        outputs = self.discriminator(fake_images)

        fake_loss = self.loss_fn(outputs, fake_labels)

        d_loss = real_loss + fake_loss
        self.log('val_discriminator_loss', d_loss, on_step=False, on_epoch=True)

        # Оценка потерь генератора
        outputs = self.discriminator(fake_images)
        g_loss = self.loss_fn(outputs, real_labels)
        self.log('val_generator_loss', g_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.conf['train']['learning_rate'])
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.conf['train']['learning_rate'])
        return [optimizer_d, optimizer_g]


class Discriminator(torch.nn.Module):
    def __init__(self, conf: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.conf = conf

        self.linear1 = torch.nn.Linear(
            conf['model']['discriminator']['image_size'],
            conf['model']['discriminator']['hidden_size']
        )
        self.linear2 = torch.nn.Linear(
            conf['model']['discriminator']['hidden_size'],
            conf['model']['discriminator']['hidden_size']
        )
        self.linear3 = torch.nn.Linear(
            conf['model']['discriminator']['hidden_size'],
            1
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


class Generator(torch.nn.Module):
    def __init__(self, conf: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.conf = conf

        self.linear1 = torch.nn.Linear(
            conf['model']['generator']['latent_size'],
            conf['model']['generator']['hidden_size']
        )
        self.linear2 = torch.nn.Linear(
            conf['model']['generator']['hidden_size'],
            conf['model']['generator']['hidden_size']
        )
        self.linear3 = torch.nn.Linear(
            conf['model']['generator']['hidden_size'],
            conf['model']['generator']['image_size']
        )
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.tanh(self.linear3(x))
        return x
