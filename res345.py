class StatePoolingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, lengths):
        # x: [batch_size, seq_len, input_dim]
        # lengths: [batch_size]
        mask = torch.arange(x.size(1)).unsqueeze(0).to(x.device) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        x_masked = x * mask  # Apply mask
        pooled = torch.sum(x_masked, dim=1) / (lengths.unsqueeze(1).float() + 1e-6)  # Mean pooling
        return self.activation(self.fc(pooled))

class SpeakerClassifier(nn.Module):
    def __init__(self, input_dim, num_speakers):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_speakers)

    def forward(self, x):
        return self.fc(x)  # [batch_size, num_speakers]

class SenseVoiceSmall(nn.Module):
    def __init__(self, ..., speaker_embed_dim=256, **kwargs):
        super().__init__()
        ...
        self.state_pooling = StatePoolingLayer(input_dim=encoder_output_size, output_dim=speaker_embed_dim)
        self.speaker_classifier = SpeakerClassifier(input_dim=speaker_embed_dim, num_speakers=kwargs["num_speakers"])
        self.speaker_ignore_id = kwargs.get("speaker_ignore_id", -1)  # Ignore ID for <|Unknown_Speaker|>

    def forward(self, speech, speech_lengths, text, text_lengths, speaker_labels, **kwargs):
        ...
        # Encoder output
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths, text)

        # Speaker embedding and classification
        speaker_embedding = self.state_pooling(encoder_out, encoder_out_lens)
        speaker_logits = self.speaker_classifier(speaker_embedding)
        speaker_loss = None

        if self.training:
            valid_indices = (speaker_labels != self.speaker_ignore_id)
            if valid_indices.any():
                speaker_loss = F.cross_entropy(speaker_logits[valid_indices], speaker_labels[valid_indices])

        # Combine losses
        total_loss = loss_ctc + loss_rich
        if speaker_loss is not None:
            total_loss += speaker_loss

        return total_loss, ...


class SenseVoiceCTCDataset(torch.utils.data.Dataset):
    def __init__(self, ..., **kwargs):
        ...
        self.speakers = list(set(item["speaker"] for item in self.contents if item["speaker"] != "<|Unknown_Speaker|>"))
        self.speaker_to_id = {spk: idx for idx, spk in enumerate(self.speakers)}
        self.unknown_speaker_id = kwargs.get("speaker_ignore_id", -1)

    def __getitem__(self, index):
        ...
        speaker = item["speaker"]
        speaker_label = self.speaker_to_id.get(speaker, self.unknown_speaker_id)

        output = {
            "speech": speech[0, :, :],
            "speech_lengths": speech_lengths,
            "text": text,
            "text_lengths": text_lengths,
            "speaker_label": torch.tensor(speaker_label, dtype=torch.int64),
        }
        return output

    def collator(self, samples):
        ...
        speaker_labels = torch.stack([s["speaker_label"] for s in samples])
        outputs["speaker_labels"] = speaker_labels
        return outputs


class DataloaderMapStyle:
    def build_iter(self, ..., **kwargs):
        ...
        def ensure_valid_speaker(batch):
            if all(s["speaker_label"] == self.unknown_speaker_id for s in batch):
                valid_sample = random.choice(
                    [s for s in self.dataset_tr if s["speaker_label"] != self.unknown_speaker_id]
                )
                batch.append(valid_sample)
            return batch

        dataloader_tr = torch.utils.data.DataLoader(
            self.dataset_tr, collate_fn=lambda batch: self.collator(ensure_valid_speaker(batch)), ...
        )
        return dataloader_tr, dataloader_val


