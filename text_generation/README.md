# IU-XRay
1. Make sure to have a valid `kaggle.json` file
2. In `text_generation` directory: `mkdir data && mkdir weights`
3. In `text_generation/data` directory: `kaggle datasets download -d raddar/chest-xrays-indiana-university`
4. In `text_generation/weights` directory: `gdown --id 19BllaOvs2x5PLV_vlWMy4i8LapLb2j6b`
4. In `text_generation/weights` directory: `wget https://archive.org/download/pubmed2018_w2v_200D.tar/pubmed2018_w2v_200D.tar.gz`