from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Usage(Enum):
    MIN_SEQ_LENGTH = "min"
    MAX_SEQ_LENGTH = "max"
    CUSTOM = "custom"

@dataclass
class SequenceSample:
    user_id: int
    poi_seq: list[int]
    category_seq: Optional[list[int]]
    timestamp_seq: list[int]
    lat_seq: list[float]
    lon_seq: list[float]
    target_poi: int

class PoiDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class GetNextBatchScheduler:
    """Manages batch iteration for sequence data with active user rotation.
    
    This scheduler handles the complex logic of rotating users in/out of batches
    and tracking which sequences have been processed.
    """
    
    def __init__(self, dataset, usage, batch_size, custom_seq_count=None):
        self.dataset = dataset
        self.usage = usage
        self.batch_size = batch_size
        self.custom_seq_count = custom_seq_count
        
        # Track processing state
        self.next_user_idx = 0
        self.active_users = []
        self.active_user_seq = []
        self.user_permutation = []
        
        # Initialize counters for each user
        self.user_processed_count = [0] * len(dataset.users)
        
    def reset(self):
        """Reset the scheduler state for a new epoch."""
        self.next_user_idx = 0
        self.active_users = []
        self.active_user_seq = []
        
        # Initialize active users
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.dataset.users)
            self.active_users.append(i)
            self.active_user_seq.append(0)
        
        # Initialize permutation
        self.user_permutation = list(range(len(self.dataset.users)))
        
        # Reset processed counters
        self.user_processed_count = [0] * len(self.dataset.users)
    
    def shuffle_users(self):
        """Shuffle user order for next epoch during training."""
        import random
        random.shuffle(self.user_permutation)
        
        # Reset active users with shuffled order
        self.next_user_idx = 0
        self.active_users = []
        self.active_user_seq = []
        
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.dataset.users)
            self.active_users.append(self.user_permutation[i])
            self.active_user_seq.append(0)
    
    def get_next_batch(self):
        """Get the next batch of sequences.
        
        Returns:
            tuple: (seqs, times, time_slots, coords, labels, lbl_times, 
                   lbl_time_slots, lbl_coords, reset_h, active_users)
        """
        seqs = []
        times = []
        time_slots = []
        coords = []
        lbls = []
        lbl_times = []
        lbl_time_slots = []
        lbl_coords = []
        reset_h = []
        
        for i in range(self.batch_size):
            i_user = self.active_users[i]
            j = self.active_user_seq[i]
            
            # Determine max sequences for this user based on usage
            max_j = self.dataset.sequences_count[i_user]
            if self.usage == Usage.MIN_SEQ_LENGTH:
                max_j = self.dataset.min_seq_count
            elif self.usage == Usage.CUSTOM:
                max_j = min(max_j, self.custom_seq_count)
            
            # Replace user if exhausted
            if j >= max_j:
                i_user = self.user_permutation[self.next_user_idx]
                j = 0
                self.active_users[i] = i_user
                self.active_user_seq[i] = j
                self.next_user_idx = (self.next_user_idx + 1) % len(self.dataset.users)
                
                # Skip users already in active set
                while self.user_permutation[self.next_user_idx] in self.active_users:
                    self.next_user_idx = (self.next_user_idx + 1) % len(self.dataset.users)
                
                self.user_processed_count[i_user] += 1
            
            # Collect batch data
            reset_h.append(j == 0)
            seqs.append(torch.tensor(self.dataset.sequences[i_user][j]))
            times.append(torch.tensor(self.dataset.sequences_times[i_user][j]))
            time_slots.append(torch.tensor(self.dataset.sequences_time_slots[i_user][j]))
            coords.append(torch.tensor(self.dataset.sequences_coords[i_user][j]))
            lbls.append(torch.tensor(self.dataset.sequences_labels[i_user][j]))
            lbl_times.append(torch.tensor(self.dataset.sequences_lbl_times[i_user][j]))
            lbl_time_slots.append(torch.tensor(self.dataset.sequences_lbl_time_slots[i_user][j]))
            lbl_coords.append(torch.tensor(self.dataset.sequences_lbl_coords[i_user][j]))
            
            self.active_user_seq[i] += 1
        
        # Stack all tensors
        return (
            torch.stack(seqs, dim=1),
            torch.stack(times, dim=1),
            torch.stack(time_slots, dim=1),
            torch.stack(coords, dim=1),
            torch.stack(lbls, dim=1),
            torch.stack(lbl_times, dim=1),
            torch.stack(lbl_time_slots, dim=1),
            torch.stack(lbl_coords, dim=1),
            torch.tensor(reset_h),
            torch.tensor(self.active_users)
        )
    
    def __len__(self):
        """Return number of batches per epoch."""
        if self.usage == Usage.MIN_SEQ_LENGTH:
            return self.dataset.min_seq_count * (len(self.dataset.users) // self.batch_size)
        elif self.usage == Usage.MAX_SEQ_LENGTH:
            return max(self.dataset.max_seq_count, self.dataset.capacity // self.batch_size)
        elif self.usage == Usage.CUSTOM:
            return self.custom_seq_count * (len(self.dataset.users) // self.batch_size)
        raise ValueError(f"Unknown usage: {self.usage}")

def collate_fn(batch):
    poi_seq = [torch.tensor(x.poi_seq) for x in batch]
    category_seq = [
        torch.tensor(x.category_seq if x.category_seq is not None else [0] * len(x.poi_seq))
        for x in batch
    ]
    timestamp_seq = [torch.tensor(x.timestamp_seq) for x in batch]
    lat_seq = [torch.tensor(x.lat_seq, dtype=torch.float32) for x in batch]
    lon_seq = [torch.tensor(x.lon_seq, dtype=torch.float32) for x in batch]

    return {
        "user": torch.tensor([x.user_id for x in batch], dtype=torch.long),
        "poi": pad_sequence(poi_seq, batch_first=True, padding_value=0),
        "category": pad_sequence(category_seq, batch_first=True, padding_value=0),
        "timestamp": pad_sequence(timestamp_seq, batch_first=True, padding_value=0),
        "lat": pad_sequence(lat_seq, batch_first=True, padding_value=0.0),
        "lon": pad_sequence(lon_seq, batch_first=True, padding_value=0.0),
        "target": torch.tensor([x.target_poi for x in batch], dtype=torch.long),
    }