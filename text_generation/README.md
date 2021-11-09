# Training
1. Make sure to have a valid `kaggle.json` file
2. In `text_generation` directory: `mkdir data && mkdir weights && mkdir weights/attentiona`
3. In `text_generation/data` directory: `kaggle datasets download -d raddar/chest-xrays-indiana-university`
4. In `text_generation/weights` directory: `gdown --id 19BllaOvs2x5PLV_vlWMy4i8LapLb2j6b`
5. In `text_generation/weights` directory: `wget https://archive.org/download/pubmed2018_w2v_200D.tar/pubmed2018_w2v_200D.tar.gz`
6. In `text_generation/weights` directory: `tar -xf pubmed2018_w2v_200D.tar.gz`

# Testing
## In text_generation directory: 
- `gdown --id 1Od62LAbdmfcK6W8TfzNcLIUouxWZoMHn`
- `git clone https://github.com/stanfordmlgroup/VisualCheXbert.git`
- Copy this to replace content in `VisualCheXbert/requirements.txt`:

```
tokenizers==0.5.2
transformers==2.5.1
```

- `!pip install -r VisualCheXbert/requirements.txt`