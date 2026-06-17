# chebnet-kaist

ChebNet tabanli KAIST CAN saldiri tespiti: rapor + calistirilabilir kod paketi.

**Ogrenci:** Tuna Girisken (91250000319)  
**Ders:** Cizge Teorisinde Olcum Parametreleri, Ege Universitesi FBE

## Icerik

```
chebnet-kaist/
├── rapor/                       # IEEE format rapor (PDF)
├── configs/                     # YAML egitim ayarlari
├── checkpoints/                 # best_model_chebnet.pth
├── data/kaist/                  # KAIST verisi (git disi, yerel kurulum)
├── src/chebnet_kaist/           # Python paketi
└── pyproject.toml
```

## Kurulum

```bash
git clone git@github.com:tunagirisken/chebnet-kaist.git
cd chebnet-kaist
pip install -e .
```

KAIST veri seti repoda yoktur (~900 MB). Indirme ve dogrulama:

```bash
python3 -m chebnet_kaist.cli.setup_data --instructions
python3 -m chebnet_kaist.cli.setup_data --verify
```

Mevcut bir kopyadan:

```bash
python3 -m chebnet_kaist.cli.setup_data --copy-from /path/to/data --verify
```

## Egitim ve degerlendirme

```bash
python3 -m chebnet_kaist.cli.train --config configs/train_chebnet.yaml
python3 -m chebnet_kaist.cli.evaluate --config configs/train_chebnet.yaml
python3 -m chebnet_kaist.cli.evaluate --config configs/train_chebnet.yaml --full-report
```

On egitilmis model: `checkpoints/best_model_chebnet.pth`

## Paket yapisi

| Dizin | Aciklama |
|-------|----------|
| `src/chebnet_kaist/cli/` | train, evaluate, setup_data |
| `src/chebnet_kaist/models/` | ChebNet mimarisi |
| `src/chebnet_kaist/data/` | KAIST yukleme, segmentasyon, graf |
| `src/chebnet_kaist/training/` | Model-agnostik egitim dongusu |
| `src/chebnet_kaist/evaluation/` | Metrikler ve rapor sekilleri |
