import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# from MBT_new_train import train
from old_train import train
feature = "ko,species"
sort = False

for disease in ['EW-T2D', 'C-T2D', 'Obesity', "LC", "IBD"]:
    for seed in [392,412,432,452,472]:
        # train(disease, feature, model_type="MTMFTransformer",
        #             use_bottleneck=False, btn_init='normal', use_cross_atn=False,
        #             seed=seed, use_config=True, mode=1)
        # train(disease, feature, model_type="MTMFTransformer",
        #             use_bottleneck=False, btn_init='embed', use_cross_atn=False,
        #             seed=seed, use_config=True, mode=1)
        # train(disease, feature, model_type="MTMFTransformer",
        #             use_bottleneck=False, btn_init='embed', use_cross_atn=True,
        #             seed=seed, use_config=True, mode=1)
        # train(disease, feature, model_type="MTMFTransformer",
        #             use_bottleneck=False, btn_init='normal', use_cross_atn=True,
        #             seed=seed, use_config=True, mode=1)
        # train(disease, feature, model_type="MTMFTransformer",
        #             use_bottleneck=True, btn_init='normal', use_cross_atn=True,
        #             seed=seed, use_config=True, mode=1)
        # train(disease=disease, feature=feature, model_type="MTMFTransformer",
        #             use_bottleneck=True, btn_init='normal', use_cross_atn=False,
        #             seed=seed, use_config=True, mode=1)
        # train(disease, feature, model_type="MTMFTransformer",
        #             use_bottleneck=True, btn_init='embed', use_cross_atn=False,
        #             seed=seed, use_config=True, mode=1)
        if disease in ["EW-T2D", "C-T2D", "Obesity"]:
            train(disease, feature, model_type="MTMFTransformer",
                    use_bottleneck=True, btn_init='embed', use_cross_atn=True, sort=False,
                    seed=seed, use_config=True, mode=1)
        if disease == "LC":
            train(disease, feature, model_type="MTMFTransformer",
                    use_bottleneck=True, btn_init='embed', use_cross_atn=True, sort=False,
                    seed=seed, use_config=True, mode=2)
        if disease == "IBD":
            train(disease, feature, model_type="MTMFTransformer",
                    use_bottleneck=True, btn_init='embed', use_cross_atn=True, sort=False,
                    seed=seed, use_config=True, mode=1)