

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from main import Fusion_Model
from load_database import dataset_loaders
from configurations import Config
from main import weights_init
from main import custom_weights_init
import random
import numpy as np
import time
from datetime import timedelta
from sklearn.metrics import accuracy_score



start_time = time.time()

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

config = Config(dataset_name='HCI')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
train_loaders, test_loaders = dataset_loaders(config.dataset_name, batch_size=config.batch_size)

best_result = []

print(f"Number of train loaders: {len(train_loaders)}")
print(f"Number of test loaders: {len(test_loaders)}")

for p in range(config.num_subjects):
    train_loader = train_loaders[p]
    test_loader = test_loaders[p]

    acc_sum, f1_sum = 0.0, 0.0  # Initialize accumulators for accuracy and F1 score
    model = Fusion_Model(config).to(device)
    model.apply(weights_init)
    model.apply(custom_weights_init)
    model.PSD_map_backbone.init_weights()
    model.hf_icma.init_weights()

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    conv1_outputs = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for eeg_map, eeg_stat, peri, labels in train_loader:
            eeg_map, eeg_stat, peri, labels = eeg_map.to(device), eeg_stat.to(device), peri.to(device), labels.to(device)
            labels = labels[:, 1].long() if labels.dim() > 1 else labels.long()
            
            optimizer.zero_grad()
            _, _, _, _, e = model(eeg_map, eeg_stat, peri)
            loss = criterion(e, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * eeg_map.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        test_acc, all_predictions, all_labels = 0.0, [], []
        with torch.no_grad():
            for eeg_map, eeg_stat, peri, labels in test_loader:
                eeg_map, eeg_stat, peri, labels = eeg_map.to(device), eeg_stat.to(device), peri.to(device), labels.to(device)
                labels = labels[:, 1].long() if labels.dim() > 1 else labels.long()

                _, _, _, _, e = model(eeg_map, eeg_stat, peri)
                _, predictions = torch.max(e, 1)
                test_acc += torch.sum(predictions == labels.data).item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())




        test_acc = accuracy_score(all_labels, all_predictions)
        test_f1 = f1_score(all_labels, all_predictions, average='macro')
        print(f"Subject ID {p+1}, Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Acc: {test_acc*100:.2f}%, Test F1: {test_f1:.2f}")

        # conv1_outputs_array = np.concatenate(conv1_outputs, axis=0)
        # output_folder = ""
        # np.save(output_folder + f"conv1_outputs_subject_{p + 1}.npy", conv1_outputs_array)

        acc_sum += test_acc
        f1_sum += test_f1

# Display the best results
print("Average Test Results:")
for result in best_result:
    print(f"Subject {result[0]}: Avg Acc: {result[1]:.2f}%, Avg F1: {result[2]:.2f}")

# Save the results to a file
results_df = pd.DataFrame(best_result, columns=['Subject ID', 'Average Accuracy', 'Average F1 Score'])
results_df.to_excel("", index=False)


end_time = time.time()
elapsed_time = end_time - start_time
formatted_time = str(timedelta(seconds=elapsed_time))
print(f"Total runtime: {formatted_time}")
