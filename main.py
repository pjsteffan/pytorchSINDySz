from datasets import WRsmallepoch
from model import SINDySz, SINDyModel

from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.utils.data as data
import torch
import lightning as L



def main(data_file, annotation_file, sample_rate=5000):
    
    dataset = WRsmallepoch(
        data_file = data_file, 
        annotation_file = annotation_file,
        single_channel_flag=False,
        psd_flag=False,
        epoch_id_restriction=2.0,
        epoch_size= 5.0, 
        sample_rate= sample_rate
    )
    
    
    trv_set_size = int(len(dataset) * 0.8)
    #test_set_size = len(dataset) - trv_set_size

    trv_indices = list(range(trv_set_size))
    test_indices = list(range(trv_set_size, len(dataset)))


    trv_set = data.Subset(dataset, trv_indices)
    test_set = data.Subset(dataset, test_indices)
    
    # use 20% of training data for validation
    train_set_size = int(len(trv_set) * 0.8)
    valid_set_size = len(trv_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(trv_set, [train_set_size, valid_set_size], generator=seed)




    train_loader = DataLoader(train_set, batch_size=9)
    valid_loader = DataLoader(valid_set, batch_size=9)
    test_loader = DataLoader(test_set, batch_size=9)
    
    time_dim = 500
    system_features = 2
    latent_features = 4
    poly_order = 2
    batch_size = 2
    
    model = SINDyModel(time_dim, system_features, latent_features,poly_order)
    # Ensure model uses default dtype for consistency with loaded data
    model = model.to(torch.get_default_dtype())
    sindy_sz = SINDySz(model)

    trainer = L.Trainer(max_epochs=10,log_every_n_steps=10, accelerator="gpu", devices=1)
    trainer.fit(sindy_sz, train_loader, valid_loader)
    #trainer.test(sindy_sz, dataloaders=test_loader)

if __name__ == "__main__":
    main('/app/Data/WR/WR5_Run4.hdf5', '/app/Data/WR/Annotations/260218_annotations_a.pkl')
