Usage

Training

The pipeline consist of two stages, the pose transfer method and conditioned upsampling method. We utilize SPGNet as the baseline method of pose transfer.  We utilize c2_matching as the conditioned upsampling method. 

first stage
bash train_personHD_jitter.sh
bash train_personHD_finetune.sh

second stage
train_personHD_upsampling.sh


Test

After training, to test with default setting, simply run:
first stage
bash test_personHD.sh

second stage
bash test_personHD_upsampling.sh




Citation
Please consider citing our paper in your publications if the project helps your research.
Tong X,Song C,Zhang Z. “PersonHD: Towards High Definition Person Image Generation” Under review.

Acknowledgement
Our code is partially based on Pose-Transfer
,SPGNet,and C2-Matching. We thank the authors for sharing their code.
