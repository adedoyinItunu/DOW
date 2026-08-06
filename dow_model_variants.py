"""
dow_model_variants.py  --  architecture controls for the artefact chapter
========================================================================
Three architectures, all trained under the identical protocol so that any
difference is attributable to the architecture alone.

  minimal   the registered model (dow_model.DoWNetCNN), 5,028 params.
            Two conv blocks, global average pooling, linear head.
            Because GAP is followed directly by a linear layer, the Grad-CAM
            channel weights are independent of spatial position and Grad-CAM
            reduces exactly to CAM (thesis 3.6.1).

  nogap     identical convolutional trunk, but the feature map is pooled once
            more and flattened into the head instead of globally averaged.
            Position is retained, so Grad-CAM no longer collapses to CAM.
            Use --width 16 for the parameter-matched variant (~5,236 params
            against minimal's 5,028) so that capacity is not a confound.

  downet    three conv layers + four dense layers, following the configuration
            Kelly et al. describe for DoWNet. Exact layer widths are not
            published; the widths here are chosen so that the saved model is
            close to the 231 kB Kelly et al. report, which is the only
            quantitative fact published about the model's size. Record this in
            the thesis as an implementation choice, not as a specification.

Print parameter counts before running anything:

    python dow_model_variants.py
"""
import torch
import torch.nn as nn


class NoGapCNN(nn.Module):
    """Same trunk as DoWNetCNN, but the head sees position.

    width=32 mirrors the registered model's channel count (~10,276 params).
    width=16 is the parameter-matched control (~5,236 params).
    """

    def __init__(self, n_classes=4, width=16, in_hw=(24, 30)):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(                       # <- Grad-CAM target
            nn.Conv2d(16, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
        )
        # the only structural change: pool + flatten instead of GAP
        self.pool = nn.MaxPool2d(2)
        with torch.no_grad():
            d = self.pool(self.block2(self.block1(torch.zeros(1, 1, *in_hw))))
            flat = d.flatten(1).shape[1]
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(flat, n_classes)

    def forward(self, x):
        x = self.block2(self.block1(x))
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


class DoWNetFull(nn.Module):
    """Three conv layers, four dense layers, as described for DoWNet.

    Widths are a reconstruction, not a specification. Check
    `param_report()` against the 231 kB figure and adjust `c` / `d` if you
    want a closer match, then state the calibration in the methodology.
    """

    def __init__(self, n_classes=4, c=(8, 16, 32), d=(64, 32, 16), in_hw=(24, 30)):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.block1 = blk(1, c[0])
        self.blockA = blk(c[0], c[1])
        self.block2 = blk(c[1], c[2])          # <- Grad-CAM target, named to
                                               #    match the existing scripts
        with torch.no_grad():
            f = self.block2(self.blockA(self.block1(torch.zeros(1, 1, *in_hw))))
            flat = f.flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat, d[0]), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(d[0], d[1]), nn.ReLU(),
            nn.Linear(d[1], d[2]), nn.ReLU(),
            nn.Linear(d[2], n_classes),
        )

    def forward(self, x):
        x = self.block2(self.blockA(self.block1(x)))
        return self.head(x.flatten(1))


def build(arch, n_classes=4, width=16):
    """Factory used by architecture_controls.py."""
    if arch == "minimal":
        from dow_model import DoWNetCNN
        return DoWNetCNN(n_classes=n_classes)
    if arch == "nogap":
        return NoGapCNN(n_classes=n_classes, width=width)
    if arch == "downet":
        return DoWNetFull(n_classes=n_classes)
    raise ValueError(f"unknown arch: {arch}")


def param_report():
    print(f"{'architecture':<22}{'params':>10}{'float32 size':>16}")
    print("-" * 48)
    for name, m in [
        ("minimal (registered)", build("minimal")),
        ("nogap width=16", build("nogap", width=16)),
        ("nogap width=32", build("nogap", width=32)),
        ("downet reconstruction", build("downet")),
    ]:
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{name:<22}{n:>10,}{n * 4 / 1024:>13.1f} kB")
    print("\nDoWNet as published: 231 kB saved model (Kelly et al. 2024),")
    print("which is roughly 57,000 float32 parameters.")


if __name__ == "__main__":
    param_report()
