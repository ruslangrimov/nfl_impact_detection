python ./lib/scripts/train_stage1.py -cfg ./lib/scripts/configs/train/v_0.py -e v_0_f0 -f 0 -w 4
python ./lib/scripts/train_stage2.py -cfg ./lib/scripts/configs/train/v_00.py -e v_00_f0 -f 0 -w 4
cp ./checkpoints/v_00_f0_e1.pth ./models/model0.pth

python ./lib/scripts/train_stage1.py -cfg ./lib/scripts/configs/train/v_1.py -e v_1_f0 -f 0 -w 4
python ./lib/scripts/train_stage2.py -cfg ./lib/scripts/configs/train/v_10.py -e v_10_f0 -f 0 -w 4
cp ./checkpoints/v_10_f0_e1.pth ./models/model1.pth

python ./lib/scripts/train_stage1.py -cfg ./lib/scripts/configs/train/v_2.py -e v_2_f0 -f 0 -w 4
python ./lib/scripts/train_stage2.py -cfg ./lib/scripts/configs/train/v_20.py -e v_20_f0 -f 0 -w 4
python ./lib/scripts/merge_heads.py -i v_20_f0_e2.pth v_20_f0_e3.pth -o model2.pth

python ./lib/scripts/train_stage1.py -cfg ./lib/scripts/configs/train/v_3.py -e v_3_f0 -f 0 -w 4
python ./lib/scripts/train_stage2.py -cfg ./lib/scripts/configs/train/v_30.py -e v_300_f0 -f 0 -w 4
python ./lib/scripts/train_stage2.py -cfg ./lib/scripts/configs/train/v_30.py -e v_301_f0 -f 0 -w 4
python ./lib/scripts/train_stage2.py -cfg ./lib/scripts/configs/train/v_31.py -e v_31_f0 -f 0 -w 4
python ./lib/scripts/merge_heads.py -i v_300_f0_e3.pth v_301_f0_e3.pth v_31_f0_e5.pth v_31_f0_e11.pth -o model3.pth

python ./lib/scripts/train_helm_model.py -cfg ./lib/scripts/configs/train/helm/v1.py -e helm_v1_f0 -f 0 -w 2
cp ./checkpoints/helm_v1_f0_e11.pth ./models/helm_v1_f0_e11.pth

