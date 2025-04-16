import torch
from torch.nn.utils.rnn import pad_sequence
from s3prl.nn.upstream import S3PRLUpstream
from s3prl.upstream.interfaces import UpstreamBase

class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt: str = "wav2vec2_xlsr", **kwargs):
        super().__init__(**kwargs)

        # Load the upstream model using the updated interface
        self.model = S3PRLUpstream(name=ckpt)
        self.model.eval()  # Set to eval mode

        # Add a hook to the last encoder output
        self.add_hook("model.model.encoder", lambda input, output: output[0])

    def forward(self, wavs):
        """
        Args:
            wavs (List[Tensor]): List of 1D FloatTensors
        Returns:
            dict: model features
        """
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        padded_wav = pad_sequence(wavs, batch_first=True).to(device)

        # Forward through the model
        features = self.model(padded_wav, wav_lengths)  # returns a dict
        return features  # contains keys like "default", "hidden_states", etc.
