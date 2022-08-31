import torch as tr
from matplotlib import pyplot as plt
from torch import nn

from sr_invariant_conv1d import SRInvariantConv1d


def experiments() -> None:
    sr_1 = 16000
    # sr_1 = 22050
    # sr_1 = 44100
    # sr_1 = 48000

    points = tr.linspace(0, 2 * tr.pi * 3, sr_1)
    audio = tr.sin(points)
    plt.title(f"sr_1: {sr_1}")
    plt.plot(audio)

    kernel_n = 69
    kernel = tr.linspace(-1.0, 1.0, kernel_n).view(1, 1, -1)

    conv = nn.Conv1d(1, 1, kernel_size=(kernel_n,), dilation=(1,))
    with tr.no_grad():
        conv.weight = nn.Parameter(kernel)

    out = conv(audio.view(1, 1, -1))
    plt.plot(out.view(-1).detach())
    plt.show()

    # sr_2 = 16000
    # sr_2 = 22050
    sr_2 = 44100
    # sr_2 = 48000
    # sr_2 = 88200

    points_2 = tr.linspace(0, 2 * tr.pi * 3, sr_2)
    audio_2 = tr.sin(points_2)
    plt.title(f"sr_2: {sr_2}")
    plt.plot(audio_2)

    out_2 = conv(audio_2.view(1, 1, -1))
    plt.plot(out_2.view(-1).detach())

    sri_conv = SRInvariantConv1d(conv, sr_1, sr_2)
    out_3 = sri_conv.forward(audio_2.view(1, 1, -1))
    plt.plot(out_3.view(-1).detach())
    plt.show()


if __name__ == "__main__":
    experiments()
