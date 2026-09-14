"""Trace tensor shapes through the DAC pipeline for one real batch.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=5 python scripts/debug_trace_shapes.py --config conf/base_cw.yml
    CUDA_VISIBLE_DEVICES=5 python scripts/debug_trace_shapes.py --config conf/base_cw_eeg.yml

Builds the model / discriminator / losses / dataset exactly as train_cw.py does
(via argbind + the yml), pulls one batch, and records shapes and value stats at
every stage: dataset -> transform -> encoder blocks -> RVQ -> decoder blocks ->
discriminator fmaps -> STFT / mel loss internals.

Output is written to scripts/debug_trace_shapes_<config stem>.out
(e.g. conf/base_cw_eeg.yml -> scripts/debug_trace_shapes_base_cw_eeg.out).
Pass --stdout to print to the terminal instead.
"""
import argparse
import contextlib
import math
import sys
from pathlib import Path

import argbind
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_cw as T  # reuses the same argbind bindings as training
from audiotools import AudioSignal
from audiotools.core import util


def stats(t: torch.Tensor):
    t = t.detach().float()
    return f"min={t.min():+.3e} max={t.max():+.3e} mean={t.mean():+.3e} std={t.std():.3e}"


def sh(t):
    return "x".join(str(s) for s in t.shape)


def _conv_info(mod):
    if isinstance(mod, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        return f"  [k={mod.kernel_size[0]} s={mod.stride[0]} d={mod.dilation[0]} p={mod.padding[0]}]"
    return ""


def hook_sequential(seq, title, depth=0):
    """Recursively attach hooks printing in/out shapes for every module in a block.

    Descends into EncoderBlock / DecoderBlock / ResidualUnit (their inner
    nn.Sequential) so the dilated convs and Snakes inside are shown too.
    Weight-normed convs are shown as plain Conv1d / ConvTranspose1d.
    """
    handles = []
    indent = "    " + "  " * depth
    for name, child in seq.named_children():
        tname = type(child).__name__
        label = f"{title}.{name}"
        if tname in ("EncoderBlock", "DecoderBlock", "ResidualUnit"):
            inner = child.block
            extra = ""
            if tname == "ResidualUnit":
                d = [m for m in inner.modules() if isinstance(m, torch.nn.Conv1d)][0].dilation[0]
                extra = f" (dilation={d}, out = x + block(x))"
            def make_block(label, tname, extra, child):
                def h(mod, inp, out):
                    print(f"{indent}{label:<12} {tname + extra:<44} {sh(inp[0]):>14} -> {sh(out)}")
                return h
            def make_header(label, tname, extra):
                def pre(mod, inp):
                    print(f"{indent}{label:<12} {tname + extra}:")
                return pre
            handles.append(child.register_forward_pre_hook(make_header(label, tname, extra)))
            handles += hook_sequential(inner, label, depth + 1)
            handles.append(child.register_forward_hook(make_block(label, tname, extra, child)))
        else:
            def make(label, tname):
                def h(mod, inp, out):
                    print(f"{indent}{label:<12} {tname:<44} {sh(inp[0]):>14} -> {sh(out):<14}{_conv_info(mod)}")
                return h
            handles.append(child.register_forward_hook(make(label, tname)))
    return handles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--duration", type=float, default=None, help="override train/AudioDataset.duration")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--stdout", action="store_true", help="print to terminal instead of the .out file")
    a = p.parse_args()

    if a.stdout:
        return run(a)
    out_path = Path(__file__).resolve().parent / f"debug_trace_shapes_{Path(a.config).stem}.out"
    with open(out_path, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        run(a)
    print(f"wrote {out_path}")


def run(a):

    args = argbind.load_args(a.config)
    if a.duration is not None:
        args["train/AudioDataset.duration"] = a.duration
    args["args.debug"] = True
    dev = a.device
    torch.manual_seed(0)

    with argbind.scope(args):
        gen = T.DAC().to(dev).eval()
        disc = T.Discriminator().to(dev).eval()
        stft_loss = T.losses.MultiScaleSTFTLoss()
        mel_loss = T.losses.MelSpectrogramLoss()
        wav_loss = T.losses.L1Loss()
        gan_loss = T.losses.GANLoss(disc)
        with argbind.scope(args, "train"):
            ds = T.build_dataset(gen.sample_rate)

    sr = gen.sample_rate
    hop = int(gen.hop_length)
    print("=" * 100)
    print(f"CONFIG {a.config}")
    print(f"  model sample_rate={sr} Hz  hop_length={hop} samples  -> token rate {sr/hop:.2f} Hz  latent_dim={gen.latent_dim}")
    print(f"  encoder_rates={gen.encoder_rates} decoder_rates={gen.decoder_rates}  n_codebooks={gen.n_codebooks} codebook_size={gen.codebook_size} codebook_dim={gen.codebook_dim}")
    print(f"  delay (samples)={gen.delay}  generator params={sum(p.numel() for p in gen.parameters())/1e6:.1f}M  disc params={sum(p.numel() for p in disc.parameters())/1e6:.1f}M")
    # receptive field of encoder in seconds (approx via delay*2)
    print(f"  approx receptive field ~ {2*gen.delay} samples = {2*gen.delay/sr:.3f} s")

    # ---------------- dataset ----------------
    print("\n[1] DATASET / LOADER")
    dur = args.get("train/AudioDataset.duration", args.get("AudioDataset.duration"))
    print(f"  train duration={dur}s  -> {int(dur*sr)} samples at {sr} Hz; EEG loader={args.get('AudioLoader.EEG', False)}")
    items = [ds[i] for i in range(a.batch_size)]
    s0 = items[0]["signal"]
    print(f"  one item: audio_data {sh(s0.audio_data)} sr={s0.sample_rate}  path={Path(str(s0.path_to_file)).name}")
    print(f"           raw values: {stats(s0.audio_data)}")
    batch = ds.collate(items)
    batch = util.prepare_batch(batch, dev)
    sig = batch["signal"]
    print(f"  collated batch: audio_data {sh(sig.audio_data)}  (B x C x T)")
    print(f"  transform_args keys: {list(batch['transform_args'].keys())[:8]} ...")

    with torch.no_grad():
        sig = ds.transform(sig.clone(), **batch["transform_args"])
    print(f"  after transform: {sh(sig.audio_data)}  {stats(sig.audio_data)}")
    x = sig.audio_data

    # ---------------- generator ----------------
    print("\n[2] GENERATOR: preprocess (right-pad to hop multiple)")
    T_in = x.shape[-1]
    right_pad = math.ceil(T_in / hop) * hop - T_in
    print(f"  T={T_in} -> padded {T_in + right_pad} (right_pad={right_pad})  n_frames={ (T_in+right_pad)//hop }")

    print("\n[3] ENCODER  (B x 1 x T -> B x latent_dim x T/hop)")
    hs = hook_sequential(gen.encoder.block, "enc")
    with torch.no_grad():
        xp = gen.preprocess(x, sr)
        z_e = gen.encoder(xp)
    for h in hs: h.remove()
    print(f"  encoder out z: {sh(z_e)}  {stats(z_e)}")

    print("\n[4] RVQ  (residual VQ over T/hop frames)")
    with torch.no_grad():
        zq, codes, latents, c_loss, cb_loss = gen.quantizer(z_e)
    print(f"  z_q {sh(zq)}  codes {sh(codes)} (B x n_codebooks x frames, ints in [0,{gen.codebook_size}))  latents {sh(latents)} (B x n_codebooks*codebook_dim x frames)")
    print(f"  commitment={c_loss.item():.4f} codebook={cb_loss.item():.4f}")
    for i in range(codes.shape[1]):
        print(f"    codebook {i}: unique codes used in batch = {codes[:, i].unique().numel():4d} / {gen.codebook_size}   (over {codes[:, i].numel()} frames)")
    print(f"  bitrate: {gen.n_codebooks} books * log2({gen.codebook_size})={math.log2(gen.codebook_size):.0f} bits * {sr/hop:.2f} frames/s = {gen.n_codebooks*math.log2(gen.codebook_size)*sr/hop:.1f} bit/s")

    print("\n[5] DECODER  (B x latent_dim x frames -> B x 1 x T)")
    hs = hook_sequential(gen.decoder.model, "dec")
    with torch.no_grad():
        y = gen.decoder(zq)
    for h in hs: h.remove()
    y = y[..., :T_in]
    print(f"  decoder out (cropped to input length): {sh(y)}  {stats(y)}   <- note final Tanh: output in [-1,1]")
    recons = AudioSignal(y, sr)

    # ---------------- discriminator ----------------
    print("\n[6] DISCRIMINATOR (input after preprocess: DC removed, peak-normalised to 0.8)")
    with torch.no_grad():
        xd = disc.preprocess(x)
        print(f"  disc.preprocess: {stats(xd)}")
        for d in disc.discriminators:
            name = type(d).__name__
            if name == "MPD":
                Tp = x.shape[-1]
                padn = d.period - Tp % d.period
                print(f"  MPD(period={d.period}): T={Tp} pad {padn} -> reshape to (B,1,{(Tp+padn)//d.period},{d.period})   [reflect pad requires pad < T]")
            elif name == "MRD":
                wl = d.window_length
                hl = d.stft_params.hop_length
                n_bins = wl // 2 + 1
                print(f"  MRD(window={wl}, hop={hl}): n_freq_bins={n_bins}  bin width={sr/wl:.3f} Hz  bands(bins)={d.bands}")
                if wl > x.shape[-1]:
                    print(f"      !! window {wl} > signal length {x.shape[-1]}: STFT is mostly zero-padding")
            try:
                fmaps = d(xd)
                print("      fmaps: " + ", ".join(sh(f) for f in fmaps))
            except Exception as e:
                print(f"      !! FAILED: {type(e).__name__}: {e}")

    # ---------------- losses ----------------
    print("\n[7] LOSSES (recons vs signal)")
    with torch.no_grad():
        print(f"  waveform L1 = {wav_loss(recons, sig).item():.4f}")
        print("  MultiScaleSTFTLoss:")
        for s in stft_loss.stft_params:
            m = sig.stft(s.window_length, s.hop_length, s.window_type).abs()
            print(f"    window={s.window_length:5d} hop={s.hop_length:4d}: magnitude {sh(m)} (B x C x freq x frames) bin width={sr/s.window_length:.3f} Hz, Nyquist={sr/2} Hz")
        print(f"    total = {stft_loss(recons, sig).item():.4f}")
        print("  MelSpectrogramLoss:")
        import numpy as np
        for n_mels, fmin, fmax, s in zip(mel_loss.n_mels, mel_loss.mel_fmin, mel_loss.mel_fmax, mel_loss.stft_params):
            mel = sig.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax,
                                      window_length=s.window_length, hop_length=s.hop_length, window_type=s.window_type)
            fb = AudioSignal.get_mel_filters(sr=sr, n_fft=s.window_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
            empty = int((fb.sum(1) == 0).sum())
            below_eps = float((mel < mel_loss.clamp_eps).float().mean())
            print(f"    window={s.window_length:5d} n_mels={n_mels:4d}: mel {sh(mel)}  empty mel filters={empty}/{n_mels}  frac of mel values below clamp_eps({mel_loss.clamp_eps})={below_eps:.2f}")
        print(f"    total = {mel_loss(recons, sig).item():.4f}")
        try:
            g, f = gan_loss.generator_loss(recons, sig)
            print(f"  GAN gen_loss={g.item():.4f} feat_loss={f.item():.4f}")
        except Exception as e:
            print(f"  GAN loss FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
