Model Zoo
---
path:checkpoints/FlowReg/
best_net_netF.pth

链接：https://pan.baidu.com/s/1-gP8oI8iqk9xScriPpbUDA 
提取码：6mjw

path:checkpoints/PoseTransfer_personHD_finetune_jitter_2e5_front
40_net_netG.pth

链接：https://pan.baidu.com/s/1Ocgb0ICu5w85zlAOxGjXFg 
提取码：pma6

path:upsample_pretrained_model
feature_extraction.pth

链接：https://pan.baidu.com/s/182Lsfici6y6tPffxXJFOKQ 
提取码：gr9t

path:upsample_experiments/stage3_restoration_gan_personHD_front_512_unresize1_finetune
net_g_100000.pth

链接：https://pan.baidu.com/s/1hq1qwwvQfy3_HsFsjHCB0g 
提取码：zpvv


Inference 
---

Test with default setting, simply run:

```bash
#first stage
bash test_personHD.sh
```

```bash
#second stage
bash test_personHD_upsampling.sh//second stage
```

Training
---

The pipeline consist of two stages, the pose transfer method and conditioned upsampling method. We utilize SPGNet as the baseline method of pose transfer.  We utilize c2_matching as the conditioned upsampling method. 

```bash
#first stage
bash train_personHD_jitter.sh
bash train_personHD_finetune.sh
```

```bash
#second stage
train_personHD_upsampling.sh
```





Citation
---
Please consider citing our paper in your publications if the project helps your research.
Tong X,Song C,Zhang Z. “PersonHD: Towards High Definition Person Image Generation” Under review.

Acknowledgement
---
Our code is partially based on 
,[SPGNet](https://github.com/cszy98/SPGNet "悬停显示") and [C2-Matching](https://github.com/yumingj/C2-Matching "悬停显示"). We thank the authors for sharing their code.

Contact
---
If you have any question, please feel free to contact us via tongxueqing2020@ia.ac.cn
