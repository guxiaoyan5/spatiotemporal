from torch.utils.data import DataLoader

from dataset.HazeDataset import HazeData


def get_dataloader(config, batch_size):
    results = {}
    train_data = HazeData(config, flag='Train')
    val_data = HazeData(config, flag='Val')
    test_data = HazeData(config, flag='Test')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    results["train_loader"] = train_loader
    results["val_loader"] = val_loader
    results["test_loader"] = test_loader
    results['pm25_mean'] = train_data.pm25_mean
    results['pm25_std'] = train_data.pm25_std
    results['feature_mean'] = train_data.feature_mean
    results['feature_std'] = train_data.feature_std
    results['wind_mean'] = train_data.wind_mean
    results['wind_std'] = train_data.wind_std
    return results
